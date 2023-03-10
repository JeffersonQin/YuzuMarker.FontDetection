import traceback
import click
import requests
import os
import sqlite3
from lxml import html
import concurrent.futures
from typing import List, Tuple
import io
from tqdm import tqdm


num_workers = 100

# init paths
root_dir = os.path.dirname(os.path.realpath(__file__))
cache_dir = os.path.join(root_dir, "cache")
artist_page_cache_path = os.path.join(cache_dir, "uta-net.db")
name_list_url_prefix = "https://www.uta-net.com/name_list/"
url_prefix = "https://www.uta-net.com"
artist_url_prefix = "https://www.uta-net.com/artist/"
song_url_prefix = "https://www.uta-net.com/song/"

artist_page_suffixes = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    39,
    40,
    41,
    42,
    43,
    44,
    70,
]

if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

# init db
def init_artist_page_db():
    conn = sqlite3.connect(artist_page_cache_path)
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS `index` (index_id INTEGER PRIMARY KEY)""")
    c.execute(
        """CREATE TABLE IF NOT EXISTS artists (artist_id INTEGER PRIMARY KEY, artist_name TEXT, done INTEGER DEFAULT 0)"""
    )
    c.execute("""CREATE TABLE IF NOT EXISTS songs (song_id INTEGER PRIMARY KEY)""")
    c.execute(
        """CREATE TABLE IF NOT EXISTS lyrics (song_id INTEGER PRIMARY KEY, lyrics TEXT)"""
    )
    conn.commit()


init_artist_page_db()


# scraper abstraction
class Scraper:
    def __init__(self, desc: str, num_workers: int, func, workload: List):
        self.desc = desc
        self.num_workers = num_workers
        self.func = func
        self.workload = workload

    def run(self):
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            _ = list(
                tqdm(
                    executor.map(self.func, self.workload),
                    total=len(self.workload),
                    leave=True,
                    desc=self.desc,
                    miniters=1,
                ),
            )


# db ops
def has_index_done_db(index_id):
    conn = sqlite3.connect(artist_page_cache_path)
    c = conn.cursor()
    c.execute(f"SELECT index_id FROM `index` WHERE index_id=?", (index_id,))
    row = c.fetchone()
    if row is not None:
        return True
    return False


def complete_index_page_db(index_id, artists):
    conn = sqlite3.connect(artist_page_cache_path)
    c = conn.cursor()
    c.execute(
        f"INSERT OR REPLACE INTO `index` VALUES (?)",
        (index_id,),
    )

    for artist_id, artist_name in artists:
        c.execute(
            f"INSERT INTO artists(artist_id, artist_name) SELECT ?, ? WHERE NOT EXISTS (SELECT 1 FROM artists WHERE artist_id=?)",
            (artist_id, artist_name, artist_id),
        )

    conn.commit()


def get_all_artists_db():
    conn = sqlite3.connect(artist_page_cache_path)
    c = conn.cursor()
    c.execute(f"SELECT artist_id, artist_name FROM artists")
    rows = c.fetchall()
    return rows


def has_artist_done_db(artist_id):
    conn = sqlite3.connect(artist_page_cache_path)
    c = conn.cursor()
    c.execute(
        f"SELECT artist_id FROM artists WHERE artist_id=? AND done=1", (artist_id,)
    )
    row = c.fetchone()
    if row is not None:
        return True
    return False


def complete_artist_page_db(artist_id, songs):
    conn = sqlite3.connect(artist_page_cache_path)
    c = conn.cursor()
    c.execute(f"UPDATE artists SET done=? WHERE artist_id=?", (1, artist_id))

    for song_id in songs:
        c.execute(f"INSERT OR REPLACE INTO songs VALUES (?)", (song_id,))

    conn.commit()


def get_all_songs_db():
    conn = sqlite3.connect(artist_page_cache_path)
    c = conn.cursor()
    c.execute(f"SELECT song_id FROM songs")
    rows = c.fetchall()
    rows = [row[0] for row in rows]
    return rows


def has_lyrics_db(song_id):
    conn = sqlite3.connect(artist_page_cache_path)
    c = conn.cursor()
    c.execute(f"SELECT song_id FROM lyrics WHERE song_id=?", (song_id,))
    row = c.fetchone()
    if row is not None:
        return True
    return False


def update_lyrics_db(song_id, lyrics):
    conn = sqlite3.connect(artist_page_cache_path)
    c = conn.cursor()
    c.execute(f"INSERT OR REPLACE INTO lyrics VALUES (?, ?)", (song_id, lyrics))
    conn.commit()


# scraping ops
def scrape_artist_list(index_id):
    # Make a GET request to the URL
    response = requests.get(f"{name_list_url_prefix}{index_id}")

    # Parse the HTML content of the response
    tree = html.fromstring(response.text)

    # Find all of the links and names using the given XPath pattern
    links = tree.xpath(
        f'//*[contains(@id,"anchor_")]/dl/dd/ul//li/p[@class="flex-glow"]/a/@href'
    )
    names = tree.xpath(f'//*[contains(@id,"anchor_")]/dl/dd/ul//li/p/a/text()')

    # Convert the links to integers
    ids = []
    for link in links:
        ids.append(int(link.replace("artist", "").replace("/", "")))

    ret = []
    ret.extend(zip(ids, names))

    return ret


def scrape_song_list(artist_id):
    # Make a GET request to the URL
    response = requests.get(f"{artist_url_prefix}{artist_id}")

    # Parse the HTML content of the response
    tree = html.fromstring(response.text)

    # Find all of the links using the given XPath pattern
    links = tree.xpath(
        f'//*[@id="list-song"]/div[2]/div[1]/div[2]/div[2]//table/tbody//tr/td[1]/a/@href'
    )

    # Convert the links to integers
    ids = []
    for link in links:
        try:
            ids.append(int(link.replace("song", "").replace("/", "")))
        except:
            pass

    return ids


def scrape_lyrics(song_id):
    # Make a GET request to the URL
    response = requests.get(f"{song_url_prefix}{song_id}")

    # Parse the HTML content of the response
    tree = html.fromstring(response.text)

    # Find all of the links using the given XPath pattern
    song_lyrics = tree.xpath('//*[@id="kashi_area"]/text()')

    return "\n".join(song_lyrics)


@click.command()
@click.option(
    "--no-cache-index",
    is_flag=True,
    default=False,
    help="Do not use cached index page",
)
@click.option(
    "--no-cache-artist",
    is_flag=True,
    default=False,
    help="Do not use cached artist page",
)
@click.option(
    "--no-cache-lyrics", is_flag=True, default=False, help="Do not use cached lyrics"
)
def scrape(no_cache_index, no_cache_artist, no_cache_lyrics):
    # artists
    def scrape_artist(index_id):
        while True:
            try:
                if not no_cache_index and has_index_done_db(index_id):
                    pass
                else:
                    artists = scrape_artist_list(index_id)
                    complete_index_page_db(index_id, artists)

                break
            except Exception as e:
                traceback.print_exc()
                print(f"Error: {e}")
                print("Retrying ...")
                continue

    Scraper(
        desc="Scraping artists",
        num_workers=num_workers,
        func=scrape_artist,
        workload=artist_page_suffixes,
    ).run()

    artists = get_all_artists_db()
    for artist_id, name in artists:
        print(f"Detected: {name} -> {artist_id}")
    print(f"{len(artists)} artists detected")

    # songs
    def scrape_song(artist: Tuple[int, str]):
        while True:
            try:
                artist_id, name = artist
                if not no_cache_artist and has_artist_done_db(artist_id):
                    print(f"Skipping: {name} -> {artist_id}")
                else:
                    print(f"Scraping: {name} -> {artist_id}")
                    songs = scrape_song_list(artist_id)
                    complete_artist_page_db(artist_id, songs)

                break
            except Exception as e:
                traceback.print_exc()
                print(f"Error: {e}")
                print("Retrying ...")
                continue

    Scraper(
        desc="Scraping songs",
        num_workers=num_workers,
        func=scrape_song,
        workload=artists,
    ).run()

    songs = get_all_songs_db()

    # lyrics
    def scrape_lyric(song_id):
        while True:
            try:
                if not no_cache_lyrics and has_lyrics_db(song_id):
                    pass
                else:
                    lyrics = scrape_lyrics(song_id)
                    update_lyrics_db(song_id, lyrics)
                break
            except Exception as e:
                traceback.print_exc()
                print(f"Error: {e}")
                print("Retrying ...")
                continue

    Scraper(
        desc="Scraping lyrics",
        num_workers=num_workers,
        func=scrape_lyric,
        workload=songs,
    ).run()


if __name__ == "__main__":
    scrape()
