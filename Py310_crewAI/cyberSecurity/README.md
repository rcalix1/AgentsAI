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
