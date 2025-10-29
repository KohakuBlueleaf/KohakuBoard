# KohakuBoard - Frontend

Modern experiment tracking dashboard with high-performance visualizations.

## Features

- Vue 3 + ElementPlus + UnoCSS
- WebGL-based plots (100K+ datapoints)
- Configurable dashboards with drag-and-drop
- Multiple visualization types:
  - Line plots with advanced smoothing (EMA, MA, Gaussian)
  - Media viewer (images/videos)
  - Histograms with step navigation
  - Tables with image column support
- Persistent layout (localStorage)
- Responsive design (mobile-friendly)

## License

**Kohaku Software License 1.0**

This is a premium feature of KohakuHub with commercial usage restrictions.

- ✅ Free for non-commercial use  
- ⚠️ Commercial trial: 3 months OR $25k revenue/year
- ⚠️ After trial, requires commercial license

Contact: kohaku@kblueleaf.net

## Development

```bash
npm install
npm run dev
```

Visit: http://localhost:5177

## Build

```bash
npm run build
```

Output: `dist/`

## Scripts

- `npm run dev` - Dev server with hot reload
- `npm run build` - Production build  
- `npm run format` - Format code with prettier
- `npm run prebuild` - Copy logos (auto-runs before build)
