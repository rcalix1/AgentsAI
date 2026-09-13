## Cyber Security with AI agents

* on windows
* cmd
* ollama run llama3.2
* 


## Putty, Virtual Box and linux VM

* Virtual Box Linux setup for Windows Host:
* Latest Putty if on Windows or SSH on MAC
* Linux VM -> Seed Labs
* Virtual Box
* For NAT, configure this in VirtualBox:



```


VM Settings → Network → Adapter 1 → NAT → Advanced → Port Forwarding

Name:       SSH
Protocol:   TCP
Host IP:    127.0.0.1
Host Port:  2222
Guest IP:   10.0.2.15
Guest Port: 22

Then on Putty

Host: 127.0.0.1
Port: 2222
SSH

Important: don't put 10.0.2.15 into PuTTY when using this NAT setup. Use 127.0.0.1 in Putty
  



```


## Paramiko SSH


```

!pip install paramiko

```


## Code for SSH



```




import paramiko

VM_IP = "192.168.1.123"   # replace with your VM's IP
USERNAME = "your_username"
PASSWORD = "your_password"

# This is what your CrewAI agent produced
results = "ping -c 4 8.8.8.8"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect(
    VM_IP,
    username=USERNAME,
    password=PASSWORD
)

stdin, stdout, stderr = ssh.exec_command(results)

print("COMMAND:")
print(results)

print("\nOUTPUT:")
print(stdout.read().decode())

print("ERRORS:")
print(stderr.read().decode())

ssh.close()




```






