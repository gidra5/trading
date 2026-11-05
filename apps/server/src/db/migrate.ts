import { openDatabase } from "./index.js";

const database = openDatabase();
try {
  console.log(`Database migrations applied: ${database.path}`);
} finally {
  database.close();
}
