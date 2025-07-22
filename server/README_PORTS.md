# Step-Change App Port Configuration

## Backend (FastAPI)
- **Default port**: 8000
- **To run on port 8001:**

If you use `uvicorn` directly, run:

```sh
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

If you use a process manager (like gunicorn or supervisor), set the port to 8001 in the command or config.

---

## Frontend (Vite + React)
- **Default port**: 5173
- **To run on port 5177:**

Add the `server.port` option to your `vite.config.js`:

```js
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5177,
  },
})
```

Then run:

```sh
npm run dev
# or
yarn dev
```

---

## Summary
- Backend: Start with `--port 8001`
- Frontend: Add `server.port = 5177` in `vite.config.js`
