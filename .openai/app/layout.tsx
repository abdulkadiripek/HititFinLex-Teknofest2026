import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HititFinLex | Katılım Finans Karar Platformu",
  description:
    "Katılım bankalarının finansman ürünlerini keşfedin, koşulları karşılaştırın ve her bilgiyi resmî kaynağında doğrulayın.",
  applicationName: "HititFinLex",
  openGraph: {
    title: "HititFinLex | Katılım Finans Karar Platformu",
    description:
      "Konut, taşıt, ihtiyaç ve ticari finansman seçeneklerini kaynaklı yapay zekâ ile karşılaştırın.",
    locale: "tr_TR",
    type: "website",
  },
  twitter: {
    card: "summary_large_image",
    title: "HititFinLex | Katılım Finans Karar Platformu",
    description:
      "Katılım finansına özel, kaynaklı ürün keşfi ve karşılaştırma deneyimi.",
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="tr" suppressHydrationWarning>
      <body>{children}</body>
    </html>
  );
}
