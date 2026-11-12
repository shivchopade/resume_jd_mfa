# ---------------------------------------------------------------------------
# Frontend image: plain HTML/CSS/JS served by nginx.
# nginx also forwards every /api/... request to the backend container,
# so the browser only ever talks to one address: http://localhost:8080
# ---------------------------------------------------------------------------
FROM nginx:stable-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY public/ /usr/share/nginx/html/

EXPOSE 80
