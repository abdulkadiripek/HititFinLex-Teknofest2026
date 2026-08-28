import type { Metadata } from "next";
import { IBM_Plex_Serif, Inter } from "next/font/google";
import "./globals.css";

const sans = Inter({
  subsets: ["latin"],
  variable: "--font-sans",
  display: "swap",
});

// Baslik ve rakam agirlikli alanlar icin: gercek 500/600 kesimleri olan,
// kurumsal/guven algisina hitap eden bir serif. Georgia'da 500 agirligi
// yoktu; tarayici bunu sahte (synthetic) kalinlastiriyordu ve ozellikle
// rakamlar boyle durumlarda garip/tutarsiz gorunuyordu.
const serif = IBM_Plex_Serif({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
  variable: "--font-serif",
  display: "swap",
});

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
    // Arayuz kucuk px olcegiyle tasarlandi; butun oraniyla birlikte buyutulur.
    // Deger globals.css'teki --ui-zoom ile ayni tutulmali (vh-tabanli
    // yukseklikler o degeri kullanarak zoom'u telafi ediyor).
    <html
      className={`${sans.variable} ${serif.variable}`}
      lang="tr"
      suppressHydrationWarning
      style={{ zoom: "var(--ui-zoom)" }}
    >
      <body>{children}</body>
    </html>
  );
}
