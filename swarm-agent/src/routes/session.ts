import { Subject } from "functools-kit";
import { app, upgradeWebSocket } from "src/config/app";
import { ioc } from "src/services";

app.get("/api/v1/session/:clientId", upgradeWebSocket((ctx) => {
  const clientId = ctx.req.param("clientId");

  const incomingSubject = new Subject<string>();

  return {
    onOpen(_, ws) {
      const receive = ioc.connectionPublicService.connect(clientId, async (outgoing) => {
        ws.send(JSON.stringify(outgoing));
      });
      incomingSubject.subscribe(receive);
    },
    onMessage(event) {
      const data = JSON.parse(event.data.toString());
      incomingSubject.next(data);
    },
    onClose: () => {
      ioc.connectionPublicService.dispose(clientId);
    },
  }
}));

export default app;
