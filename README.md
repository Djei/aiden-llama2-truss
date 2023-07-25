Project to package llama2 into a [truss image](https://github.com/basetenlabs/truss)

This requires:
- python3.8 minimum

To build:
- python3.8 build-truss.py
- docker build build -t aiden-llama2-truss:latest

To launch:
- docker run -v /home/ec2-user/llama:/llama -p 127.0.0.1:8080:8080 -t aiden-llama2-truss:latest