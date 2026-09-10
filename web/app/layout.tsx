import type { Metadata } from 'next';
import './globals.css';
export const metadata: Metadata = { title: 'PV Visual Placer', description: 'Explore Swiss roofs and fit real-size solar modules with local YOLO11 segmentation.' };
export default function RootLayout({children}: Readonly<{children: React.ReactNode}>) {
  return <html lang="en"><body>{children}</body></html>;
}
