FROM node:22-slim
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --production
COPY server.js render.js ./
EXPOSE 3456
CMD ["node", "server.js"]
