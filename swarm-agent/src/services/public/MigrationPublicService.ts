import { inject } from "src/core/di";
import LoggerService from "../base/LoggerService";
import TYPES from "src/config/types";
import MigrationPrivateService from "../private/MigrationPrivateService";
import ContextService from "../base/ContextService";

interface IMigrationPrivateService extends MigrationPrivateService {}

type TMigrationPrivateService = {
  [key in keyof IMigrationPrivateService]: any;
};

export class MigrationPublicService implements TMigrationPrivateService {
  readonly loggerService = inject<LoggerService>(TYPES.loggerService);
  readonly migrationPrivateService = inject<MigrationPrivateService>(
    TYPES.migrationPrivateService
  );

  public createProduct = async (title: string, description: string) => {
    this.loggerService.log(`migrationPublicService createProduct`, {
      title,
      description,
    });
    return await ContextService.runInContext(
      async () => {
        return await this.migrationPrivateService.createProduct(
          title,
          description
        );
      },
      {
        clientId: "migration-service",
      }
    );
  };

  public findProduct = async (search: string) => {
    this.loggerService.log(`migrationPublicService findProduct`, {
      search,
    });
    return await ContextService.runInContext(
      async () => {
        return await this.migrationPrivateService.findProduct(search);
      },
      {
        clientId: "migration-service",
      }
    );
  };

  public importProducts = async (path: string) => {
    this.loggerService.log(`migrationPublicService importProducts`, {
      path,
    });
    return await ContextService.runInContext(
      async () => {
        return await this.migrationPrivateService.importProducts(path);
      },
      {
        clientId: "migration-service",
      }
    );
  };
}

export default MigrationPublicService;
