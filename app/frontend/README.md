# ViHSD Frontend (Next.js)

Comment Moderation Studio UI. Consumes the FastAPI backend.

## Setup
```
cd app/frontend
npm install
cp .env.local.example .env.local   # set NEXT_PUBLIC_API_URL to your backend
npm run dev                        # http://localhost:3000
```

## Routes
- `/` — Studio: predict + explain + showdown + rewrite
- `/simulate` — batch CSV scoring dashboard
- `/insights` — model metrics table

## Test
```
npm run test       # vitest component/lib tests
npm run build      # production build check
```

Backend must be running at `NEXT_PUBLIC_API_URL` for live data.
