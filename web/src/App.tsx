import { Chat } from "./components/Chat";
import { StatusBar } from "./components/StatusBar";
import { Telemetry } from "./components/Telemetry";
import { useTelemetrySocket } from "./hooks/useTelemetrySocket";

export function App() {
  useTelemetrySocket();
  return (
    <div className="app">
      <div className="workspace">
        <Chat />
        <Telemetry />
      </div>
      <StatusBar />
    </div>
  );
}
