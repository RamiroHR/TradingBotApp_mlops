FROM ubuntu:20.04

ADD ./docker_files/users_credentials_api/requirements.txt ./docker_files/users_credentials_api/users_db_api.py ./

RUN apt update && apt install python3-pip -y && pip install -r requirements.txt

EXPOSE 7000 

# CMD uvicorn users_db_api:server --host 0.0.0.0
CMD ["uvicorn", "users_db_api:server", "--host", "0.0.0.0", "--port", "7000"]