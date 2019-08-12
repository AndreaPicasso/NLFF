from fabric.api import *
 
env.use_ssh_config = False
env.hosts = ["155.69.151.167"]
env.user = "senticteam"
#env.key_filename = "/root/.ssh/id_rsa"
env.password = "busysentic"
#env.port = 22
 
 
def uptime():
    print(run("sudo -s"))
    with settings(prompts={'[sudo] password for senticteam:': 'busysentic'}):
    	run("cd subj_dec/codes_mar17")
    	print(run("perl runall.pl"))
    	with settings(prompts={'Enter Choice \n1. Sentence \n2. File Name ': '1'}):
    		with settings(prompts={'Enter your input: ': 'I am a good guy'}):
    			print('qualcosa')


