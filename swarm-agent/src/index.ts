import "./config/setup";

import "./test";

import { serve } from '@hono/node-server'
import { isMainThread } from 'worker_threads';
import { serveStatic } from '@hono/node-server/serve-static'

import "./routes/session";
import "./routes/complete";

import { app, injectWebSocket } from './config/app';
import { ioc } from './services';

import { CC_EXECUTE_TEST, CC_WWWROOT_PORT } from "./config/params";

app.use('/*', serveStatic({ root: './public' }))

const main = () => {
  if (!isMainThread) {
    return;
  }

  if (CC_WWWROOT_PORT === -1) {
    return;
  }

  if (CC_EXECUTE_TEST) {
    return;
  }

  const server = serve({
    fetch: app.fetch,
    port: CC_WWWROOT_PORT,
  });
  
  server.addListener("listening", () => {
    console.log("Server listening on http://local.dev:80");
  });
  
  injectWebSocket(server)  
};

main();

ioc.loggerService.setDebug(!CC_EXECUTE_TEST);
