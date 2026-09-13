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


## Notes

```

Installing collected packages: invoke, cffi, pynacl, paramiko
  Attempting uninstall: cffi
    Found existing installation: cffi 1.17.1
    Uninstalling cffi-1.17.1:
      Successfully uninstalled cffi-1.17.1
Successfully installed cffi-2.1.1 invoke-3.0.3 paramiko-5.0.0 pynacl-1.6.2

WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))': /simple/paramiko/
WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ProtocolError('Connection aborted.', ConnectionResetError(10054, 'An existing connection was forcibly closed by the remote host', None, 10054, None))': /simple/paramiko/
  WARNING: Failed to remove contents in a temporary directory 'C:\Users\user1\AppData\Local\Temp\pip-uninstall-8seyuc9w'.
  You can safely remove it manually.
C:\Users\user1\anaconda3\envs\py310_agentsAI\lib\site-packages\IPython\utils\_process_win32.py:124: ResourceWarning: unclosed file <_io.BufferedWriter name=3>
  return process_handler(cmd, _system_body)
ResourceWarning: Enable tracemalloc to get the object allocation traceback
C:\Users\user1\anaconda3\envs\py310_agentsAI\lib\site-packages\IPython\utils\_process_win32.py:124: ResourceWarning: unclosed file <_io.BufferedReader name=4>
  return process_handler(cmd, _system_body)
ResourceWarning: Enable tracemalloc to get the object allocation traceback
C:\Users\user1\anaconda3\envs\py310_agentsAI\lib\site-packages\IPython\utils\_process_win32.py:124: ResourceWarning: unclosed file <_io.BufferedReader name=5>
  return process_handler(cmd, _system_body)
ResourceWarning: Enable tracemalloc to get the object allocation traceback




```



