import { WebSocket } from "ws";

export class WebClient {
  constructor(
    private readonly socket: WebSocket,
    private readonly onError: (error: Error) => void,
  ) {
    socket.on("error", onError);
  }

  private send(type: string, message?: unknown) {
    if (this.socket.readyState !== WebSocket.OPEN) return false;
    this.socket.send(JSON.stringify({ type, message }), (error) => {
      if (error) this.onError(error);
    });
    return true;
  }

  botStatus() {
    this.send("botStatus");
  }

  connectionStatus() {
    this.send("connectionStatus");
  }

  marketStatus() {
    this.send("marketStatus");
  }

  backtestStatus() {
    this.send("backtestStatus");
  }

  accountStatus() {
    this.send("accountStatus");
  }
}
