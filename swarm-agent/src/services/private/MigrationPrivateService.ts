import { inject } from "src/core/di";
import LoggerService from "../base/LoggerService";
import TYPES from "src/config/types";
import ProductDbService from "../db/ProductDbService";
import fs from "fs/promises";

export class MigrationPrivateService {
  private readonly loggerService = inject<LoggerService>(TYPES.loggerService);
  private readonly productDbService = inject<ProductDbService>(
    TYPES.productDbService
  );

  public createProduct = async (title: string, description: string) => {
    this.loggerService.logCtx("migrationPrivateService createProduct", {
      title,
      description,
    });
    return await this.productDbService.create({
      title,
      description,
    });
  };

  public findProduct = async (search: string) => {
    this.loggerService.logCtx("migrationPrivateService findProduct", {
      search,
    });
    return await this.productDbService.findByVector(search);
  };

  public importProducts = async (path: string) => {
    this.loggerService.logCtx("migrationPrivateService importProducts", {
      path,
    });
    const text = await fs.readFile(path);
    const products = <{ title: string; description: string }[]>(
      JSON.parse(text.toString())
    );
    for (const { title, description } of products) {
      console.log(`Creating ${title}`);
      await this.productDbService.create({
        title,
        description,
      });
    }
  };
}

export default MigrationPrivateService;
