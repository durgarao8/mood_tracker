import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DSA RAG Chatbot",
  description: "Data Structures and Algorithms RAG Chatbot",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}