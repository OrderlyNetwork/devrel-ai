FROM node:20-slim AS builder
WORKDIR /usr/src/app
COPY package.json yarn.lock tsconfig.json ./
COPY knowledge.json ./
RUN yarn install --frozen-lockfile --production=false
COPY src ./src
RUN yarn build

FROM node:20-slim AS runner
WORKDIR /usr/src/app
COPY --from=builder /usr/src/app/dist ./dist
COPY --from=builder /usr/src/app/package.json ./package.json
COPY --from=builder /usr/src/app/yarn.lock ./yarn.lock
COPY knowledge.json ./
RUN yarn install --frozen-lockfile --production=true
CMD ["yarn", "start"]