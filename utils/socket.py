from socket import *


def listen_socket(data_socket: int):
    port = data_socket
    buffer_size = 512
    listen_socket = socket(AF_INET, SOCK_STREAM)
    listen_socket.bind(('127.0.0.1', port))

    listen_socket.listen(1)
    data_socket, address = listen_socket.accept()
    info = ''
    while True:
        received = data_socket.recv(buffer_size)
        if not received:
            break
        info += received.decode()
    data_socket.close()
    listen_socket.close()
    return info


def send_socket(ip: int, server_port: int):
    data_socket = socket(AF_INET, SOCK_STREAM)
    data_socket.connect((ip, server_port))
    while True:
        to_send = input('>> ')
        if to_send == '':
            break
        data_socket.send(to_send.encode())
    data_socket.close()
