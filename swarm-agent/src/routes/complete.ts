import { errorData, getErrorMessage } from "functools-kit";
import { app } from "src/config/app";
import CompleteRequest from "src/model/CompleteRequest.model";
import { ioc } from "src/services";

app.post("/api/v1/complete", async (ctx) => {
  const request = await ctx.req.json<CompleteRequest>();
  console.time(`${ctx.req.url} ${request.requestId}`);
  ioc.loggerService.log(ctx.req.url, { request });
  try {
    const result = await ioc.connectionPublicService.complete(request.clientId, request.messages);
    ioc.loggerService.log(`${ctx.req.url} ok`, { request, result });
    return ctx.json(result, 200);
  } catch (error) {
    ioc.loggerService.log(`${ctx.req.url} error`, {
      request,
      error: errorData(error),
    });
    return ctx.json(
      {
        status: "error",
        error: getErrorMessage(error),
        clientId: request.clientId,
        requestId: request.requestId,
        serviceName: request.serviceName,
      },
      500
    );
  } finally {
    console.timeEnd(`${ctx.req.url} ${request.requestId}`);
  }
});
