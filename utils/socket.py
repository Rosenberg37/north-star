from socket import *


def listen_socket(data_socket):
    PORT = data_socket
    BUFFER = 512
    listenSocket = socket(AF_INET, SOCK_STREAM)
    listenSocket.bind(('127.0.0.1', PORT))

    listenSocket.listen(1)
    dataSocket, addr = listenSocket.accept()
    info = ''
    while True:
        received = dataSocket.recv(BUFFER)
        if not received:
            break
        info += received.decode()
    dataSocket.close()
    listenSocket.close()
    return info


def send_socket(IP, SERVER_PORT):
    dataSocket = socket(AF_INET, SOCK_STREAM)
    dataSocket.connect((IP, SERVER_PORT))
    while True:
        toSend = input('>> ')
        if toSend == '':
            break
        dataSocket.send(toSend.encode())
    dataSocket.close()
