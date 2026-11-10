# ---------- 构建阶段 ----------
FROM node:22-alpine AS build

WORKDIR /app

# 先只拷贝依赖清单，利用 Docker 层缓存：改业务代码不会触发重装依赖
COPY package.json package-lock.json ./
RUN npm ci

COPY . .
RUN npm run build

# ---------- 运行阶段 ----------
FROM nginx:1.27-alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
