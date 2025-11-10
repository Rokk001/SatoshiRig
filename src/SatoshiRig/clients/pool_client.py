import json
import socket
import logging
from typing import Tuple , List


class PoolClient :
    def __init__(self , host: str , port: int , timeout: int = 30) :
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.logger = logging.getLogger("SatoshiRig.pool_client")

    def connect(self) :
        try:
            self.sock = socket.socket(socket.AF_INET , socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host , self.port))
        except (socket.error, OSError, ConnectionError) as e:
            self.logger.error(f"Failed to connect to {self.host}:{self.port}: {e}")
            raise

    def subscribe(self) -> Tuple[str , str , int] :
        assert self.sock is not None
        try:
            self.sock.sendall(b'{"id": 1, "method": "mining.subscribe", "params": []}\n')
            data = self.sock.recv(1024)
            if not data:
                raise ConnectionError("Connection closed by server during subscribe")
            lines = data.decode('utf-8', errors='replace').split('\n')
            response = json.loads(lines[0])
            subscription_details , extranonce1 , extranonce2_size = response['result']
            return subscription_details , extranonce1 , int(extranonce2_size)
        except (socket.error, OSError, ConnectionError, json.JSONDecodeError, KeyError, ValueError) as e:
            self.logger.error(f"Subscribe failed: {e}")
            raise

    def authorize(self , wallet_address: str) :
        assert self.sock is not None
        try:
            authorize_msg = json.dumps({
                "params": [wallet_address , "password"] ,
                "id": 2 ,
                "method": "mining.authorize"
            }).encode('utf-8') + b"\n"
            self.sock.sendall(authorize_msg)
        except (socket.error, OSError, ConnectionError) as e:
            self.logger.error(f"Authorize failed: {e}")
            raise

    def read_notify(self) -> list:
        assert self.sock is not None
        # Robust line-buffered read with simple framing by newlines
        # Keep reading until we see at least one mining.notify message
        # and we have consumed a line ending.
        buffer = bytearray()
        messages: list[str] = []
        self.sock.settimeout(self.timeout)
        try:
            while True:
                chunk = self.sock.recv(4096)
                if not chunk:
                    raise ConnectionError("Connection closed by server during read_notify")
                buffer.extend(chunk)
                while True:
                    try:
                        newline_index = buffer.index(10)  # '\n'
                    except ValueError:
                        break
                    # Use 'replace' instead of 'ignore' to preserve data integrity
                    line = buffer[:newline_index].decode('utf-8', errors='replace').strip()
                    del buffer[:newline_index + 1]
                    if line:
                        messages.append(line)
                # Stop once we have at least one notify message
                if any('mining.notify' in m for m in messages):
                    break
        except (socket.timeout, socket.error, OSError, ConnectionError) as e:
            self.logger.error(f"read_notify failed: {e}")
            raise
        responses = []
        for m in messages:
            try:
                obj = json.loads(m)
                if 'mining.notify' in m:
                    responses.append(obj)
            except json.JSONDecodeError as e:
                self.logger.debug(f"Failed to parse message: {m[:100]}... Error: {e}")
                continue
        return responses

    def submit(self , wallet_address: str , job_id: str , extranonce2: str , ntime: str , nonce_hex: str) -> bytes:
        assert self.sock is not None
        try:
            payload = json.dumps({
                "params": [wallet_address , job_id , extranonce2 , ntime , nonce_hex] ,
                "id": 1 ,
                "method": "mining.submit"
            }).encode('utf-8') + b"\n"
            self.sock.sendall(payload)
            response = self.sock.recv(1024)
            if not response:
                raise ConnectionError("Connection closed by server during submit")
            return response
        except (socket.error, OSError, ConnectionError) as e:
            self.logger.error(f"Submit failed: {e}")
            raise


