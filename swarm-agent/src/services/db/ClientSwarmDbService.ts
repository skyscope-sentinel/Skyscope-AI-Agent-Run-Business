import BaseMap from "src/common/BaseMap";
import { CC_CLIENT_SESSION_EXPIRE_SECONDS } from "src/config/params";
import { obsolete } from "functools-kit";

export class ClientSwarmDbService extends BaseMap(
  "node-ollama-agent-swarm__clientSwarmDb",
  {
    TTL_EXPIRE_SECONDS: CC_CLIENT_SESSION_EXPIRE_SECONDS,
  }
) {
  setWithKeepExpire = obsolete(
    async (key: string, value: any): Promise<void> => {
      this.loggerService.logCtx(
        `clientSwarmDbService setWithKeepExpire key=${key} connection=${this.connectionKey}`,
        { key, value }
      );
      return await super.setWithKeepExpire(key, value);
    },
    "clientSwarmDbService setWithKeepExpire"
  );

  set = async (key: string, value: any): Promise<void> => {
    this.loggerService.logCtx(
      `clientSwarmDbService set key=${key} connection=${this.connectionKey}`,
      { key, value }
    );
    return await super.set(key, value);
  };

  get = async (key: string): Promise<any | null> => {
    this.loggerService.logCtx(
      `clientSwarmDbService get key=${key} connection=${this.connectionKey}`
    );
    return await super.get(key);
  };

  delete = async (key: string): Promise<void> => {
    this.loggerService.logCtx(
      `clientSwarmDbService delete key=${key} connection=${this.connectionKey}`
    );
    return Promise.resolve();
  };

  has = obsolete(async (key: string): Promise<boolean> => {
    this.loggerService.logCtx(
      `clientSwarmDbService has key=${key} connection=${this.connectionKey}`
    );
    return await super.has(key);
  }, "clientSwarmDbService has");

  clear = async (): Promise<void> => {
    this.loggerService.logCtx(
      `clientSwarmDbService clear connection=${this.connectionKey}`
    );
    return Promise.resolve();
  };

  getFirst = obsolete(async (): Promise<any | null> => {
    this.loggerService.logCtx(
      `clientSwarmDbService getFirst connection=${this.connectionKey}`
    );
    return await super.getFirst();
  }, "clientSwarmDbService getFirst");

  shift = obsolete(async (): Promise<any | null> => {
    this.loggerService.logCtx(
      `clientSwarmDbService shift connection=${this.connectionKey}`
    );
    return await super.shift();
  }, "clientSwarmDbService shift");

  size = obsolete(async (): Promise<number> => {
    this.loggerService.logCtx(
      `clientSwarmDbService size connection=${this.connectionKey}`
    );
    return await super.size();
  }, "clientSwarmDbService size");
}

export default ClientSwarmDbService;
