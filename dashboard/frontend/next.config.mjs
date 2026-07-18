/** @type {import('next').NextConfig} */
const nextConfig = {
  // Static export so the FastAPI backend can serve the built UI from
  // dashboard/frontend/out (see backend/main.py). Remove `output` to use
  // the Next.js dev/prod server instead.
  output: "export",
  images: { unoptimized: true },
  reactStrictMode: true,
};

export default nextConfig;
