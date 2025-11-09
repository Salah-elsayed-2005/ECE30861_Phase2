import express from 'express';
import packageRoutes from './api/routes/packageRoutes';

const app = express();

app.use('/', packageRoutes);

export default app;