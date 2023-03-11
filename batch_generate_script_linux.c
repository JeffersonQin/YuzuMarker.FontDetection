#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>

#define MAX_DIGIT 10


int total_mission = 64;
int min_mission = 33;
int max_mission = 48;


int main(int argc, char* argv[]) {
	for (int i = min_mission; i <= max_mission; i ++) {
		int pid = fork();
		if (pid < 0) {
			perror("fork");
		}
		if (pid == 0) {
			char batch_number[MAX_DIGIT];
			char batch_count[MAX_DIGIT];

			memset(batch_number, '\0', MAX_DIGIT * sizeof(char));
			memset(batch_count, '\0', MAX_DIGIT * sizeof(char));

			sprintf(batch_number, "%d", i);
			sprintf(batch_count, "%d", total_mission);

			char *cmd = "./venv/bin/python";
			char *args[] = {"./venv/bin/python", "font_ds_generate_script.py", batch_number, batch_count, NULL};

			if (execvp(cmd, args) < 0) {
				perror("execvp");
			}
		}
	}

	pid_t wpid;
	int status = 0;
	while ((wpid = wait(&status)) > 0) {}
	return 0;
}

