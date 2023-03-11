#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
#include <string.h>

#define MAX_DIGIT 10


int total_mission = 64;
int min_mission = 33;
int max_mission = 48;

#ifndef TOTAL_MISSION
#define TOTAL_MISSION total_mission
#endif

#ifndef MIN_MISSION
#define MIN_MISSION min_mission
#endif

#ifndef MAX_MISSION
#define MAX_MISSION max_mission
#endif


int main(int argc, char* argv[]) {
	for (int i = MIN_MISSION; i <= MAX_MISSION; i ++) {
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
			sprintf(batch_count, "%d", TOTAL_MISSION);

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

