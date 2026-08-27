"use client";

import {
  FormEvent,
  ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import panoramaStyles from "./AssistantPanorama.module.css";

const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

type View = "overview" | "catalog" | "compare" | "assistant" | "quality";
type ConnectionState = "checking" | "online" | "degraded" | "offline";
type OverviewScope = "all" | "live" | "history";
type CatalogScope = "all" | "live" | "history";
type CompareScope = "live" | "history";
type ComparisonView = "cards" | "matrix";
type PanoramaSort = "coverage" | "bank" | "confidence" | "maturity" | "amount";
type HistoryPeriod = "1m" | "3m" | "6m" | "1y" | "all";
type IconName =
  | "home"
  | "search"
  | "compare"
  | "spark"
  | "chart"
  | "database"
  | "building"
  | "document"
  | "filter"
  | "arrow"
  | "external"
  | "check"
  | "clock"
  | "shield"
  | "menu"
  | "close"
  | "send"
  | "chevron"
  | "refresh"
  | "sort"
  | "layers"
  | "warning";

type HealthResponse = {
  status: string;
  model_ready: boolean;
  ner_model_ready?: boolean;
  classifier_ready?: boolean;
  ollama_model_ready?: boolean;
  document_count: number;
  chunk_count: number;
  comparison_fact_count?: number;
  gpu?: string | null;
  ollama_model?: string;
};

type DashboardBucket = {
  code: string;
  label: string;
  count: number;
  percentage: number;
};

type LatestDocument = {
  document_id: number;
  bank_name: string;
  page_title: string | null;
  campaign_type_code: string | null;
  campaign_type: string | null;
  confidence: number | null;
  source_url: string | null;
  updated_at: string | null;
};

type DashboardOverview = {
  document_count: number;
  bank_count: number;
  verified_count: number;
  fact_count: number;
  documents_with_facts: number;
  coverage_percentage: number;
  average_confidence: number;
  pending_document_reviews: number;
  pending_fact_reviews: number;
  banks: DashboardBucket[];
  product_types: DashboardBucket[];
  fact_types: DashboardBucket[];
  latest_documents: LatestDocument[];
  live_document_count?: number;
  historical_document_count?: number;
  total_snapshot_count?: number;
  history_start_date?: string | null;
  history_end_date?: string | null;
};

type HistoryCountBucket = {
  name?: string | null;
  code?: string | null;
  count: number;
};

type HistoricalOverview = {
  historical_document_count: number;
  searchable_document_count: number;
  review_document_count: number;
  historical_fact_count: number;
  historical_chunk_count: number;
  embedded_chunk_count: number;
  history_start_date: string | null;
  history_end_date: string | null;
  banks: HistoryCountBucket[];
  product_types: HistoryCountBucket[];
};

type CampaignTypeOption = {
  code: string;
  label: string;
  document_count: number;
  bank_count: number;
};

type ComparisonOptions = {
  campaign_types: CampaignTypeOption[];
  banks: string[];
  entity_labels: { code: string; label: string; entity_count: number }[];
};

type CatalogItem = {
  document_id: number;
  bank_name: string;
  page_title: string | null;
  source_url: string | null;
  campaign_type_code: string | null;
  campaign_type: string | null;
  summary_text: string | null;
  confidence: number | null;
  verified: boolean;
  fact_count: number;
  fact_types: string[];
  updated_at: string | null;
};

type CatalogResponse = {
  total: number;
  page: number;
  page_size: number;
  page_count: number;
  items: CatalogItem[];
};

type ComparisonValue = {
  text: string;
  normalized_value: Record<string, unknown> | null;
  source: string;
  confidence: number | null;
  evidence_text: string | null;
};

type ComparisonItem = {
  document_id: number;
  bank_name: string;
  page_title: string | null;
  source_url: string | null;
  campaign_type_code: string;
  campaign_type: string | null;
  summary_text: string | null;
  confidence: number | null;
  attributes: Record<string, ComparisonValue[]>;
};

type ComparisonResponse = {
  campaign_type_code: string;
  campaign_type: string | null;
  count: number;
  items: ComparisonItem[];
};

type ComparisonColumn = ComparisonItem & {
  document_count: number;
  document_ids: number[];
  page_titles: string[];
};

type DocumentFact = {
  fact_type: string;
  label: string;
  text: string;
  normalized_value: Record<string, unknown> | null;
  source: string;
  confidence: number | null;
  evidence_text: string | null;
};

type DocumentDetail = {
  document_id: number;
  bank_name: string;
  page_title: string | null;
  source_url: string | null;
  campaign_type_code: string | null;
  campaign_type: string | null;
  summary_text: string | null;
  raw_text: string;
  confidence: number | null;
  verified: boolean;
  updated_at: string | null;
  facts: DocumentFact[];
};

type ChatSource = {
  source_id: number;
  document_id?: number;
  archive_key?: string;
  bank_name: string;
  page_title: string | null;
  source_url: string | null;
  archive_url?: string | null;
  snapshot_date?: string | null;
  product_type_code?: string | null;
  content: string;
  semantic_score: number;
  lexical_score: number;
  hybrid_score: number;
};

type ChatResponse = {
  query: string;
  answer: string;
  model: string;
  sources: ChatSource[];
};

type HistoricalSearchResult = {
  rank: number;
  document_id: number;
  archive_key: string;
  bank_name: string;
  page_title: string | null;
  source_url: string | null;
  archive_url: string | null;
  snapshot_date: string | null;
  product_type_code: string | null;
  content: string;
  semantic_score: number;
  lexical_score: number;
  hybrid_score: number;
};

type HistoricalSearchResponse = {
  query: string;
  count: number;
  results: HistoricalSearchResult[];
};

type HistoricalChatResponse = {
  query: string;
  answer: string;
  model: string;
  sources: HistoricalSearchResult[];
};

type HistoricalComparisonItem = {
  document_id: number;
  archive_key: string;
  bank_name: string;
  page_title: string | null;
  source_url: string | null;
  archive_url: string | null;
  snapshot_date: string | null;
  product_type_code: string | null;
  classification_confidence: number | null;
  attributes: Record<string, ComparisonValue[]>;
};

type HistoricalComparisonResponse = {
  product_type_code: string;
  as_of: string | null;
  count: number;
  items: HistoricalComparisonItem[];
};

type ConversationItem = ChatResponse & {
  id: number;
  scope: CompareScope;
  periodLabel?: string;
  panorama?: ComparisonResponse | null;
  panoramaProduct?: string | null;
};

const factLabels: Record<string, string> = {
  ALISVERIS_PUANI: "Alışveriş puanı",
  BASVURU_KANALI: "Başvuru kanalı",
  BASVURU_SON_TARIHI: "Son başvuru tarihi",
  BELGE_MUAFIYETI: "Belge muafiyeti",
  ERKEN_ODEME_KOSULU: "Erken ödeme",
  EKSPERTIZ_UCRETI: "Ekspertiz ücreti",
  FINANSMAN_ORANI: "Finansman oranı",
  FINANSMAN_TUTARI: "Finansman tutarı",
  GEREKLI_BELGE: "Gerekli belge",
  GEREKLI_BELGELER: "Gerekli belgeler",
  HARCAMA_ESIGI: "Harcama eşiği",
  HARCAMA_UST_LIMITI: "Harcama üst limiti",
  HEDEF_KITLE: "Hedef kitle",
  INDIRIM_ORANI: "İndirim oranı",
  INDIRIM_TUTARI: "İndirim tutarı",
  IPOTEK_TESIS_UCRETI: "İpotek tesis ücreti",
  ISLEM_ALT_LIMITI: "İşlem alt limiti",
  ISLEM_UST_LIMITI: "İşlem üst limiti",
  KAMPANYA_AVANTAJI: "Kampanya avantajı",
  KAMPANYA_BASLANGIC_TARIHI: "Başlangıç tarihi",
  KAMPANYA_BITIS_TARIHI: "Bitiş tarihi",
  KAMPANYA_SURESI: "Kampanya süresi",
  KAMPANYA_TARIH_ARALIGI: "Kampanya tarihleri",
  KAR_PAYI_ORANI: "Kâr payı oranı",
  KAR_PAYLASIM_ORANI: "Kâr paylaşım oranı",
  MASRAF_DURUMU: "Masraf durumu",
  MEVDUAT_GUVENCESI: "Fon güvencesi",
  MINIMUM_BAKIYE: "Minimum bakiye",
  DIGER_UCRET: "Diğer ücret",
  ODUL_MIKTARI: "Ödül miktarı",
  ODUL_TUTARI: "Ödül tutarı",
  ODEME_PLANI: "Ödeme planı",
  PESINAT_ORANI: "Peşinat oranı",
  PESINAT_TUTARI: "Peşinat tutarı",
  SIGORTA_KOSULU: "Sigorta koşulu",
  SIGORTA_UCRETI: "Sigorta / tekafül ücreti",
  TAHSIS_UCRETI: "Tahsis ücreti",
  TAKSIT_SAYISI: "Taksit sayısı",
  TEMINAT: "Teminat",
  UYGUNLUK_KOSULU: "Yararlanma koşulu",
  VADE_SURESI: "Vade",
  VERGI_MUAFIYETI: "Vergi muafiyeti",
};

const comparisonFactProfiles: Record<string, string[]> = {
  campaign: [
    "KAMPANYA_TARIH_ARALIGI",
    "KAMPANYA_BASLANGIC_TARIHI",
    "KAMPANYA_BITIS_TARIHI",
    "KAMPANYA_SURESI",
    "HARCAMA_ESIGI",
    "HARCAMA_UST_LIMITI",
    "INDIRIM_ORANI",
    "INDIRIM_TUTARI",
    "ODUL_TUTARI",
    "ODUL_MIKTARI",
    "ALISVERIS_PUANI",
    "TAKSIT_SAYISI",
    "HEDEF_KITLE",
    "UYGUNLUK_KOSULU",
    "KAMPANYA_AVANTAJI",
    "BASVURU_KANALI",
  ],
  finance: [
    "FINANSMAN_TUTARI",
    "FINANSMAN_ORANI",
    "VADE_SURESI",
    "KAR_PAYI_ORANI",
    "PESINAT_TUTARI",
    "PESINAT_ORANI",
    "TAHSIS_UCRETI",
    "EKSPERTIZ_UCRETI",
    "IPOTEK_TESIS_UCRETI",
    "MASRAF_DURUMU",
    "ODEME_PLANI",
    "BASVURU_KANALI",
    "GEREKLI_BELGELER",
    "GEREKLI_BELGE",
    "ERKEN_ODEME_KOSULU",
    "SIGORTA_KOSULU",
  ],
  investment: [
    "MINIMUM_BAKIYE",
    "KAR_PAYLASIM_ORANI",
    "VADE_SURESI",
    "MEVDUAT_GUVENCESI",
    "VERGI_MUAFIYETI",
    "MASRAF_DURUMU",
    "ISLEM_ALT_LIMITI",
    "ISLEM_UST_LIMITI",
    "BASVURU_KANALI",
  ],
  insurance: [
    "TEMINAT",
    "SIGORTA_UCRETI",
    "SIGORTA_KOSULU",
    "VADE_SURESI",
    "HEDEF_KITLE",
    "UYGUNLUK_KOSULU",
    "BASVURU_KANALI",
  ],
  payment: [
    "ISLEM_ALT_LIMITI",
    "ISLEM_UST_LIMITI",
    "DIGER_UCRET",
    "MASRAF_DURUMU",
    "HEDEF_KITLE",
    "BASVURU_KANALI",
  ],
  other: [
    "HEDEF_KITLE",
    "UYGUNLUK_KOSULU",
    "KAMPANYA_AVANTAJI",
    "BASVURU_KANALI",
    "MASRAF_DURUMU",
  ],
};

const demoDashboard: DashboardOverview = {
  document_count: 771,
  bank_count: 10,
  verified_count: 0,
  fact_count: 3236,
  documents_with_facts: 590,
  coverage_percentage: 76.5,
  average_confidence: 0.72,
  pending_document_reviews: 0,
  pending_fact_reviews: 0,
  banks: [
    ["emlak", "Emlak Katılım", 100],
    ["ziraat", "Ziraat Katılım", 100],
    ["albaraka", "Albaraka Türk", 99],
    ["kuveyt", "Kuveyt Türk", 99],
    ["vakif", "Vakıf Katılım", 99],
    ["dunya", "Dünya Katılım", 98],
    ["turkiye", "Türkiye Finans", 86],
    ["hayat", "Hayat Finans", 67],
    ["tom", "T.O.M. Katılım", 15],
    ["adil", "Adil Katılım", 8],
  ].map(([code, label, count]) => ({
    code: String(code),
    label: String(label),
    count: Number(count),
    percentage: Math.round((Number(count) / 771) * 1000) / 10,
  })),
  product_types: [
    ["KART", "Kart kampanyası", 253],
    ["YATIRIM_URUNU", "Yatırım ürünü", 178],
    ["FINANSMAN", "Finansman", 141],
    ["DIGER", "Diğer", 103],
    ["ALISVERIS_PUANI", "Alışveriş puanı", 34],
    ["YENI_MUSTERI", "Yeni müşteri", 22],
    ["IHTIYAC_FINANSMANI", "İhtiyaç finansmanı", 20],
    ["TASIT_FINANSMANI", "Taşıt finansmanı", 13],
    ["KONUT_FINANSMANI", "Konut finansmanı", 7],
  ].map(([code, label, count]) => ({
    code: String(code),
    label: String(label),
    count: Number(count),
    percentage: Math.round((Number(count) / 771) * 1000) / 10,
  })),
  fact_types: [
    ["KAMPANYA_SURESI", 494],
    ["VADE_SURESI", 469],
    ["TAKSIT_SAYISI", 458],
    ["HARCAMA_ESIGI", 268],
    ["MASRAF_DURUMU", 255],
    ["ALISVERIS_PUANI", 243],
    ["HEDEF_KITLE", 191],
    ["INDIRIM_ORANI", 190],
    ["ODUL_MIKTARI", 178],
    ["FINANSMAN_TUTARI", 155],
  ].map(([code, count]) => ({
    code: String(code),
    label: factLabels[String(code)] ?? String(code),
    count: Number(count),
    percentage: Math.round((Number(count) / 3236) * 1000) / 10,
  })),
  latest_documents: [],
};

const demoHistory: HistoricalOverview = {
  historical_document_count: 2580,
  searchable_document_count: 1566,
  review_document_count: 1014,
  historical_fact_count: 3982,
  historical_chunk_count: 3192,
  embedded_chunk_count: 3192,
  history_start_date: "2016-01-01",
  history_end_date: "2026-08-21",
  banks: [
    { name: "Kuveyt Türk Katılım Bankası A.Ş.", count: 492 },
    { name: "Türkiye Finans Katılım Bankası A.Ş.", count: 473 },
    { name: "Ziraat Katılım Bankası A.Ş.", count: 410 },
    { name: "Vakıf Katılım Bankası A.Ş.", count: 394 },
    { name: "Albaraka Türk Katılım Bankası A.Ş.", count: 325 },
    { name: "Türkiye Emlak Katılım Bankası A.Ş.", count: 242 },
    { name: "Hayat Finans Katılım Bankası A.Ş.", count: 185 },
    { name: "Dünya Katılım Bankası A.Ş.", count: 59 },
  ],
  product_types: [
    { code: "KART_KAMPANYASI", count: 1407 },
    { code: "DIGER_KAMPANYA", count: 467 },
    { code: "DIGER", count: 170 },
    { code: "YATIRIM_URUNU", count: 147 },
    { code: "TICARI_FINANSMAN", count: 100 },
    { code: "KATILMA_HESABI", count: 70 },
    { code: "KART_URUNU", count: 63 },
    { code: "KONUT_FINANSMANI", count: 59 },
    { code: "IHTIYAC_FINANSMANI", count: 31 },
    { code: "SIGORTA_TEKAFUL_URUNU", count: 22 },
    { code: "TASIT_FINANSMANI", count: 21 },
    { code: "DIGER_FINANSMAN", count: 19 },
    { code: "ODEME_TRANSFER_HIZMETI", count: 4 },
  ],
};

const demoCatalog: CatalogResponse = {
  total: 6,
  page: 1,
  page_size: 12,
  page_count: 1,
  items: [
    [693, "Ziraat Katılım", "Finansman İş Birlikleri", "TICARI_FINANSMAN", 0.9946],
    [720, "Ziraat Katılım", "Alışveriş Finansmanı", "IHTIYAC_FINANSMANI", 0.981],
    [723, "Ziraat Katılım", "Konut-Gayrimenkul Finansmanı", "KONUT_FINANSMANI", 0.9916],
    [609, "Vakıf Katılım", "Kentsel Dönüşüm Finansmanı", "KONUT_FINANSMANI", 0.9923],
    [393, "Emlak Katılım", "Finansman Ferdi Kaza Sigortası", "SIGORTA_TEKAFUL_URUNU", 0.9932],
    [364, "Kuveyt Türk", "Finansman Ürünleri", "TICARI_FINANSMAN", 0.9944],
  ].map(([id, bank, title, code, confidence]) => ({
    document_id: Number(id),
    bank_name: String(bank),
    page_title: String(title),
    source_url: null,
    campaign_type_code: String(code),
    campaign_type: String(code).replaceAll("_", " "),
    summary_text: "Yerel API bağlandığında bu kayıt gerçek özet ve çıkarılmış alanlarla güncellenir.",
    confidence: Number(confidence),
    verified: false,
    fact_count: 0,
    fact_types: [],
    updated_at: null,
  })),
};

const demoOptions: ComparisonOptions = {
  campaign_types: demoDashboard.product_types.map((item) => ({
    code: item.code,
    label: item.label,
    document_count: item.count,
    bank_count: 0,
  })),
  banks: demoDashboard.banks.map((item) => item.label),
  entity_labels: demoDashboard.fact_types.map((item) => ({
    code: item.code,
    label: item.label,
    entity_count: item.count,
  })),
};

const demoComparison: ComparisonResponse = {
  campaign_type_code: "KONUT_FINANSMANI",
  campaign_type: "Konut Finansmanı",
  count: 3,
  items: [
    {
      document_id: 723,
      bank_name: "Ziraat Katılım",
      page_title: "Konut-Gayrimenkul Finansmanı",
      source_url: "https://www.ziraatkatilim.com.tr/bireysel/finansman-urunleri/konut-gayrimenkul-finansmani",
      campaign_type_code: "KONUT_FINANSMANI",
      campaign_type: "Konut Finansmanı",
      summary_text: "Konut ve gayrimenkul finansmanı seçenekleri.",
      confidence: 0.9916,
      attributes: {
        ODEME_PLANI: [{ text: "Esnek ödeme planı", normalized_value: null, source: "dataset_snapshot", confidence: 0.94, evidence_text: "Esnek ödeme planı seçenekleri sunulmaktadır." }],
        VERGI_MUAFIYETI: [{ text: "KKDF ve BSMV muafiyeti imkânı", normalized_value: null, source: "dataset_snapshot", confidence: 0.91, evidence_text: "Finansman kullanımında KKDF ve BSMV muafiyeti imkânı bulunabilir." }],
      },
    },
    {
      document_id: 1,
      bank_name: "Türkiye Finans",
      page_title: "Konut Finansmanı",
      source_url: "https://www.turkiyefinans.com.tr/tr-tr/bireysel/konut-finansmani/Sayfalar/konut-finansmani.aspx",
      campaign_type_code: "KONUT_FINANSMANI",
      campaign_type: "Konut Finansmanı",
      summary_text: "İlk veya mevcut konuta yönelik finansman seçenekleri.",
      confidence: 0.97,
      attributes: {
        VADE_SURESI: [{ text: "120 ay", normalized_value: { value: 120, unit: "ay" }, source: "dataset_snapshot", confidence: 0.98, evidence_text: "Mortgage finansmanında maksimum vade süresi 120 aydır." }],
        SIGORTA_KOSULU: [{ text: "Sigortalı ve sigortasız seçenekler", normalized_value: null, source: "dataset_snapshot", confidence: 0.9, evidence_text: "Sigortalı ve sigortasız alternatifler sunulur." }],
      },
    },
    {
      document_id: 609,
      bank_name: "Vakıf Katılım",
      page_title: "Konut Finansmanı",
      source_url: "https://www.vakifkatilim.com.tr/tr/kendim-icin/finansmanlar/konut-finansmani",
      campaign_type_code: "KONUT_FINANSMANI",
      campaign_type: "Konut Finansmanı",
      summary_text: "Katılım finans prensiplerine uygun konut finansmanı.",
      confidence: 0.96,
      attributes: {
        BASVURU_KANALI: [{ text: "Web üzerinden ön başvuru", normalized_value: null, source: "dataset_snapshot", confidence: 0.95, evidence_text: "Web üzerinden ön başvuru yapılabilir." }],
      },
    },
  ],
};

const quickQuestions = [
  "Konut finansmanında en uzun vadeli seçenekleri karşılaştır",
  "KOBİ için finansman tutarı içeren ürünleri bul",
  "Taşıt finansmanındaki masraf ve vade koşulları neler?",
  "Aktif kampanyalardaki alışveriş puanlarını karşılaştır",
];

const historicalQuickQuestions = [
  "2020-2021 dönemindeki kart kampanyalarında bonus koşulları nasıldı?",
  "Son bir yılda konut finansmanı koşullarında hangi değişimler görünüyor?",
  "Geçmiş taşıt finansmanı vadelerini banka bazında karşılaştır",
  "2019 sonrası KOBİ kampanyalarında öne çıkan destekler nelerdi?",
];

const historyPeriodOptions: { value: HistoryPeriod; label: string; short: string }[] = [
  { value: "1m", label: "Son 1 ay", short: "1A" },
  { value: "3m", label: "Son 3 ay", short: "3A" },
  { value: "6m", label: "Son 6 ay", short: "6A" },
  { value: "1y", label: "Son 1 yıl", short: "1Y" },
  { value: "all", label: "Tüm arşiv", short: "Tümü" },
];

function Icon({ name, size = 20 }: { name: IconName; size?: number }) {
  const paths: Record<IconName, ReactNode> = {
    home: <><path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 10v10h13V10M9.5 20v-6h5v6"/></>,
    search: <><circle cx="11" cy="11" r="7"/><path d="m16.5 16.5 4 4"/></>,
    compare: <><path d="M8 5h12M16 2l4 3-4 3M16 19H4M8 16l-4 3 4 3"/></>,
    spark: <><path d="m12 3 1.4 4.1L17.5 8.5l-4.1 1.4L12 14l-1.4-4.1-4.1-1.4 4.1-1.4L12 3Z"/><path d="m19 14 .8 2.2L22 17l-2.2.8L19 20l-.8-2.2L16 17l2.2-.8L19 14Z"/></>,
    chart: <><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></>,
    database: <><ellipse cx="12" cy="5" rx="8" ry="3"/><path d="M4 5v6c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 11v6c0 1.7 3.6 3 8 3s8-1.3 8-3v-6"/></>,
    building: <><path d="M3 21h18M5 21V9l7-5 7 5v12"/><path d="M9 21v-6h6v6M9 10h.01M15 10h.01"/></>,
    document: <><path d="M6 2h8l4 4v16H6z"/><path d="M14 2v5h5M9 12h6M9 16h6"/></>,
    filter: <path d="M3 5h18l-7 8v6l-4 2v-8L3 5Z"/>,
    arrow: <><path d="M5 12h14M14 7l5 5-5 5"/></>,
    external: <><path d="M13 4h7v7M20 4l-9 9"/><path d="M18 13v6a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h6"/></>,
    check: <path d="m5 12 4 4L19 6"/>,
    clock: <><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></>,
    shield: <><path d="M12 3 4.5 6v5.5c0 4.8 3.1 8 7.5 9.5 4.4-1.5 7.5-4.7 7.5-9.5V6L12 3Z"/><path d="m9 12 2 2 4-4"/></>,
    menu: <><path d="M4 6h16M4 12h16M4 18h16"/></>,
    close: <><path d="m5 5 14 14M19 5 5 19"/></>,
    send: <><path d="m4 4 17 8-17 8 3-8-3-8Z"/><path d="M7 12h14"/></>,
    chevron: <path d="m9 18 6-6-6-6"/>,
    refresh: <><path d="M20 6v5h-5"/><path d="M18.5 16a8 8 0 1 1 .5-8l1 3"/></>,
    sort: <><path d="M8 5h11M8 12h8M8 19h5M3 4v16M1 18l2 2 2-2"/></>,
    layers: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/></>,
    warning: <><path d="M12 3 2.5 20h19L12 3Z"/><path d="M12 9v5M12 17h.01"/></>,
  };
  return <svg aria-hidden="true" className="icon" fill="none" height={size} viewBox="0 0 24 24" width={size}>{paths[name]}</svg>;
}

async function apiRequest<T>(
  path: string,
  init: RequestInit = {},
  timeoutMs = 20000,
): Promise<T> {
  const controller = new AbortController();
  const timer = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      ...init,
      headers: { "Content-Type": "application/json", ...(init.headers ?? {}) },
      signal: controller.signal,
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || `HTTP ${response.status}`);
    }
    return (await response.json()) as T;
  } finally {
    window.clearTimeout(timer);
  }
}

function shortBank(name: string) {
  return name
    .replace("Katılım Bankası A.Ş.", "")
    .replace("Katılım Bankası", "")
    .replace("Türkiye Emlak", "Emlak")
    .trim();
}

function initials(name: string) {
  const clean = shortBank(name);
  return clean.split(/\s+/).slice(0, 2).map((part) => part[0]).join("").toUpperCase();
}

const bankBrands = [
  { match: "albaraka", slug: "albaraka", mark: "A", label: "Albaraka Türk", color: "#008b65", accent: "#d8b24c" },
  { match: "kuveyt", slug: "kuveyt", mark: "KT", label: "Kuveyt Türk", color: "#6d3b87", accent: "#d9b24a" },
  { match: "ziraat", slug: "ziraat", mark: "ZK", label: "Ziraat Katılım", color: "#157391", accent: "#55b899" },
  { match: "vakıf", slug: "vakif", mark: "VK", label: "Vakıf Katılım", color: "#a63f4b", accent: "#e2b765" },
  { match: "vakif", slug: "vakif", mark: "VK", label: "Vakıf Katılım", color: "#a63f4b", accent: "#e2b765" },
  { match: "emlak", slug: "emlak", mark: "EK", label: "Emlak Katılım", color: "#138153", accent: "#c5a75c" },
  { match: "türkiye finans", slug: "turkiyefinans", mark: "TF", label: "Türkiye Finans", color: "#7a5538", accent: "#d59f52" },
  { match: "turkiye finans", slug: "turkiyefinans", mark: "TF", label: "Türkiye Finans", color: "#7a5538", accent: "#d59f52" },
  { match: "hayat", slug: "hayat", mark: "HF", label: "Hayat Finans", color: "#23815d", accent: "#8ac7a6" },
  { match: "dünya", slug: "dunya", mark: "DK", label: "Dünya Katılım", color: "#26778b", accent: "#d7ae55" },
  { match: "dunya", slug: "dunya", mark: "DK", label: "Dünya Katılım", color: "#26778b", accent: "#d7ae55" },
  { match: "t.o.m", slug: "tom", mark: "T", label: "T.O.M. Katılım", color: "#147a83", accent: "#5dc1b6" },
  { match: "tom", slug: "tom", mark: "T", label: "T.O.M. Katılım", color: "#147a83", accent: "#5dc1b6" },
  { match: "adil", slug: "adil", mark: "A", label: "Adil Katılım", color: "#b67a22", accent: "#e2be70" },
];

function bankBrand(name: string) {
  const normalized = name.toLocaleLowerCase("tr-TR");
  return bankBrands.find((brand) => normalized.includes(brand.match)) ?? {
    slug: "generic",
    mark: initials(name),
    label: shortBank(name),
    color: "#164f43",
    accent: "#d5ab4b",
  };
}

function BankLogo({ name, wordmark = false }: { name: string; wordmark?: boolean }) {
  const brand = bankBrand(name);
  return <span
    className={`bank-logo brand-${brand.slug}${wordmark ? " with-wordmark" : ""}`}
    style={{ "--bank-color": brand.color, "--bank-accent": brand.accent } as React.CSSProperties}
    title={shortBank(name)}
  >
    <span className="bank-logo-symbol"><i /><b>{brand.mark}</b></span>
    {wordmark && <span className="bank-logo-wordmark"><strong>{brand.label}</strong><small>Katılım finansı</small></span>}
  </span>;
}

function inferProductType(query: string) {
  const value = query.toLocaleLowerCase("tr-TR");
  if (/konut|gayrimenkul|arsa|iş yeri|is yeri|kentsel dönüşüm|kentsel donusum/.test(value)) return "KONUT_FINANSMANI";
  if (/taşıt|tasit|araç|arac|otomobil|motosiklet/.test(value)) return "TASIT_FINANSMANI";
  if (/ihtiyaç|ihtiyac|alışveriş finans|alisveris finans|eğitim finans|egitim finans|hac|umre|dayanıklı tüketim|dayanikli tuketim/.test(value)) return "IHTIYAC_FINANSMANI";
  if (/ticari|kobi|tarım|tarim|işletme|isletme|eximbank|ihracat|ithalat|akreditif|teminat mektubu/.test(value)) return "TICARI_FINANSMAN";
  if (/kart|bonus|puan|taksit|harcama/.test(value)) return "KART_KAMPANYASI";
  if (/yatırım|yatirim|fon|altın|altin|döviz|doviz/.test(value)) return "YATIRIM_URUNU";
  if (/katılma hesabı|katilma hesabi|mevduat|vadeli hesap/.test(value)) return "KATILMA_HESABI";
  if (/sigorta|tekafül|tekaful|hayat sigorta|ferdi kaza/.test(value)) return "SIGORTA_TEKAFUL_URUNU";
  if (/ödeme|odeme|transfer|havale|eft|swift/.test(value)) return "ODEME_TRANSFER_HIZMETI";
  if (/kampanya|indirim|ödül|odul/.test(value)) return "DIGER_KAMPANYA";
  return null;
}

function prioritizedFacts(query: string, productCode: string, item: ComparisonColumn) {
  const value = query.toLocaleLowerCase("tr-TR");
  const priorities: string[] = [];
  if (/vade|ay|yıl|yil/.test(value)) priorities.push("VADE_SURESI");
  if (/tutar|limit|milyon|bin tl|finansman/.test(value)) priorities.push("FINANSMAN_TUTARI", "HARCAMA_UST_LIMITI");
  if (/oran|kâr payı|kar payi/.test(value)) priorities.push("KAR_PAYI_ORANI", "FINANSMAN_ORANI", "PESINAT_ORANI");
  if (/masraf|ücret|ucret/.test(value)) priorities.push("MASRAF_DURUMU", "TAHSIS_UCRETI", "EKSPERTIZ_UCRETI", "DIGER_UCRET");
  if (/ödül|odul|bonus|puan/.test(value)) priorities.push("ODUL_TUTARI", "ODUL_MIKTARI", "ALISVERIS_PUANI");
  if (/tarih|dönem|donem|ne zaman/.test(value)) priorities.push("KAMPANYA_TARIH_ARALIGI", "KAMPANYA_BITIS_TARIHI");
  const ordered = [...new Set([...priorities, ...comparisonProfile(productCode), ...Object.keys(item.attributes)])];
  return ordered
    .filter((factType) => item.attributes[factType]?.length)
    .map((factType) => ({ factType, values: item.attributes[factType] }))
    .slice(0, 6);
}

function formatNumber(value: number) {
  return new Intl.NumberFormat("tr-TR").format(value);
}

function formatDate(value: string | null) {
  if (!value) return "—";
  return new Intl.DateTimeFormat("tr-TR", { day: "2-digit", month: "short", year: "numeric" }).format(new Date(value));
}

function isoDate(value: Date) {
  return value.toISOString().slice(0, 10);
}

function shiftDate(value: string, period: HistoryPeriod) {
  const date = new Date(`${value}T12:00:00`);
  if (period === "1m") date.setMonth(date.getMonth() - 1);
  if (period === "3m") date.setMonth(date.getMonth() - 3);
  if (period === "6m") date.setMonth(date.getMonth() - 6);
  if (period === "1y") date.setFullYear(date.getFullYear() - 1);
  return isoDate(date);
}

function periodDates(period: HistoryPeriod, history: HistoricalOverview) {
  const end = history.history_end_date ?? isoDate(new Date());
  return {
    dateFrom: period === "all" ? history.history_start_date : shiftDate(end, period),
    dateTo: end,
  };
}

function historicalBuckets(
  items: HistoryCountBucket[],
  kind: "bank" | "product",
): DashboardBucket[] {
  const total = items.reduce((sum, item) => sum + item.count, 0);
  return items.map((item) => {
    const code = kind === "bank" ? item.name ?? "unknown" : item.code ?? "UNCLASSIFIED";
    return {
      code,
      label: kind === "bank" ? item.name ?? "Bilinmeyen kurum" : friendlyCode(item.code ?? "UNCLASSIFIED"),
      count: item.count,
      percentage: total ? Math.round((item.count / total) * 1000) / 10 : 0,
    };
  });
}

function mergeDashboardBuckets(
  live: DashboardBucket[],
  history: DashboardBucket[],
  kind: "bank" | "product",
) {
  const merged = new Map<string, DashboardBucket>();
  [...live, ...history].forEach((item) => {
    const key = kind === "bank" ? shortBank(item.label).toLocaleLowerCase("tr-TR") : item.code;
    const current = merged.get(key);
    if (current) current.count += item.count;
    else merged.set(key, { ...item });
  });
  const values = [...merged.values()].sort((left, right) => right.count - left.count);
  const total = values.reduce((sum, item) => sum + item.count, 0);
  return values.map((item) => ({
    ...item,
    percentage: total ? Math.round((item.count / total) * 1000) / 10 : 0,
  }));
}

function normalizeHistoricalComparison(data: HistoricalComparisonResponse): ComparisonResponse {
  return {
    campaign_type_code: data.product_type_code,
    campaign_type: friendlyCode(data.product_type_code),
    count: data.count,
    items: data.items.map((item) => ({
      document_id: item.document_id,
      bank_name: item.bank_name,
      page_title: item.page_title,
      source_url: item.archive_url ?? item.source_url,
      campaign_type_code: item.product_type_code ?? data.product_type_code,
      campaign_type: friendlyCode(item.product_type_code ?? data.product_type_code),
      summary_text: item.snapshot_date ? `${formatDate(item.snapshot_date)} tarihli arşiv kesiti` : "Tarihsel arşiv kesiti",
      confidence: item.classification_confidence,
      attributes: item.attributes,
    })),
  };
}

function scoreLabel(value: number | null) {
  return value == null ? "—" : `%${Math.round(value * 100)}`;
}

function friendlyCode(code: string | null, label?: string | null) {
  if (label && label !== code) return label;
  if (!code) return "Etiketsiz";
  return code.toLocaleLowerCase("tr-TR").replaceAll("_", " ").replace(/(^|\s)\S/g, (char) => char.toLocaleUpperCase("tr-TR"));
}

function comparisonProfile(code: string) {
  if (code.includes("KART") || code.includes("KAMPANYA") || code === "ALISVERIS_PUANI" || code === "YENI_MUSTERI") return comparisonFactProfiles.campaign;
  if (code.includes("FINANSMAN")) return comparisonFactProfiles.finance;
  if (code.includes("YATIRIM") || code.includes("HESAP")) return comparisonFactProfiles.investment;
  if (code.includes("SIGORTA") || code.includes("TEKAFUL")) return comparisonFactProfiles.insurance;
  if (code.includes("ODEME") || code.includes("TRANSFER")) return comparisonFactProfiles.payment;
  return comparisonFactProfiles.other;
}

function aggregateComparisonItems(items: ComparisonItem[]): ComparisonColumn[] {
  const groups = new Map<string, ComparisonColumn>();
  const seen = new Map<string, Set<string>>();

  items.forEach((item) => {
    let group = groups.get(item.bank_name);
    if (!group) {
      group = {
        ...item,
        attributes: {},
        document_count: 0,
        document_ids: [],
        page_titles: [],
      };
      groups.set(item.bank_name, group);
      seen.set(item.bank_name, new Set());
    }

    group.document_count += 1;
    group.document_ids.push(item.document_id);
    if (item.page_title && !group.page_titles.includes(item.page_title)) group.page_titles.push(item.page_title);
    if ((item.confidence ?? 0) > (group.confidence ?? 0)) {
      group.document_id = item.document_id;
      group.page_title = item.page_title;
      group.source_url = item.source_url;
      group.summary_text = item.summary_text;
      group.confidence = item.confidence;
    }

    Object.entries(item.attributes).forEach(([factType, values]) => {
      const bucket = group!.attributes[factType] ?? [];
      values.forEach((value) => {
        const signature = `${factType}:${value.text.toLocaleLowerCase("tr-TR").replace(/\s+/g, " ").trim()}`;
        if (seen.get(item.bank_name)!.has(signature)) return;
        seen.get(item.bank_name)!.add(signature);
        bucket.push(value);
      });
      group!.attributes[factType] = bucket;
    });
  });

  return [...groups.values()];
}

function comparisonFactCount(item: ComparisonColumn) {
  return Object.values(item.attributes).reduce((total, values) => total + values.length, 0);
}

function visibleComparisonFacts(code: string, columns: ComparisonColumn[]) {
  const coverage = new Map<string, number>();
  columns.forEach((column) => {
    Object.entries(column.attributes).forEach(([factType, values]) => {
      if (values.length) coverage.set(factType, (coverage.get(factType) ?? 0) + 1);
    });
  });
  const profile = comparisonProfile(code);
  const preferred = profile.filter((factType) => coverage.has(factType));
  const additional = [...coverage.keys()]
    .filter((factType) => !preferred.includes(factType))
    .sort((left, right) => (coverage.get(right) ?? 0) - (coverage.get(left) ?? 0));
  return [...preferred, ...additional].slice(0, 12);
}

function renderAnswer(text: string, sources: ChatSource[]) {
  return text.split("\n").map((line, lineIndex) => {
    if (!line.trim()) return <div className="answer-space" key={`space-${lineIndex}`} />;
    const parts = line.split(/(\*\*[^*]+\*\*|\[\d+\])/g).filter(Boolean);
    const content = parts.map((part, index) => {
      const cite = part.match(/^\[(\d+)\]$/);
      if (cite && sources.some((source) => source.source_id === Number(cite[1]))) {
        return <a className="citation" href={`#source-${cite[1]}`} key={`${part}-${index}`}>{part}</a>;
      }
      if (part.startsWith("**") && part.endsWith("**")) return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
      return <span key={`${part}-${index}`}>{part}</span>;
    });
    return line.trim().startsWith("-") ? <div className="answer-line bullet" key={lineIndex}><i />{content}</div> : <p className="answer-line" key={lineIndex}>{content}</p>;
  });
}

function BankComparisonCard({
  item,
  productCode,
  query = "",
  historical = false,
  onOpen,
}: {
  item: ComparisonColumn;
  productCode: string;
  query?: string;
  historical?: boolean;
  onOpen: () => void;
}) {
  const facts = prioritizedFacts(query, productCode, item);
  const metricFacts = facts.slice(0, 3);
  const benefitFacts = facts.slice(3, 6);
  return <article className="finance-bank-card">
    <div className="finance-bank-identity">
      <BankLogo name={item.bank_name} wordmark />
      <div className="finance-bank-source">
        <span>{historical ? "TARİHSEL KESİT" : "GÜNCEL KAYNAKLAR"}</span>
        <strong>{item.page_titles[0] ?? item.page_title ?? friendlyCode(productCode)}</strong>
        <small>{item.document_count} belge · {comparisonFactCount(item)} yapılandırılmış bilgi</small>
      </div>
    </div>
    <div className="finance-bank-benefits">
      <span>ÖNE ÇIKAN KOŞULLAR</span>
      {facts.length ? (benefitFacts.length ? benefitFacts : facts).slice(0, 3).map(({ factType, values }) => <p key={factType}>
        <i><Icon name="check" size={13} /></i>
        <span><b>{factLabels[factType] ?? friendlyCode(factType)}:</b> {values.slice(0, 2).map((value) => value.text).join(" · ")}</span>
      </p>) : <p className="fact-missing"><i><Icon name="warning" size={13} /></i><span>Kaynak belgeler bulundu; karşılaştırılabilir koşul henüz yapılandırılmamış.</span></p>}
      {item.page_titles.length > 1 && <small>{item.page_titles.length} farklı ürün/kaynak başlığı banka bazında birleştirildi.</small>}
    </div>
    <div className="finance-bank-metrics">
      {metricFacts.length ? metricFacts.map(({ factType, values }) => <div key={factType}>
        <span>{factLabels[factType] ?? friendlyCode(factType)}</span>
        <strong>{values[0]?.text ?? "—"}</strong>
        {values.length > 1 && <small>+{values.length - 1} ek değer</small>}
      </div>) : <div className="metric-empty"><span>Bilgi durumu</span><strong>Kaynak mevcut</strong><small>Yapılandırılmış alan bekleniyor</small></div>}
    </div>
    <div className="finance-bank-actions">
      <span className="confidence-badge"><Icon name="shield" size={14} /> {scoreLabel(item.confidence)} güven</span>
      <button onClick={onOpen} type="button">Kaynakları incele <Icon name="arrow" size={15} /></button>
      <details>
        <summary>Tüm alanları göster ({comparisonFactCount(item)})</summary>
        <div>{Object.entries(item.attributes).map(([factType, values]) => <p key={factType}><b>{factLabels[factType] ?? friendlyCode(factType)}</b><span>{values.map((value) => value.text).join(" · ")}</span></p>)}</div>
      </details>
    </div>
  </article>;
}

function panoramaValues(item: ComparisonColumn, factTypes: string[]) {
  const values = factTypes.flatMap((factType) => item.attributes[factType] ?? []).map((value) => value.text.trim()).filter(Boolean);
  return [...new Set(values)];
}

function panoramaNumber(text: string) {
  const match = text.match(/\d[\d.]*?(?:,\d+)?(?=\s|TL|₺|%|$)/i) ?? text.match(/\d[\d.]*(?:,\d+)?/);
  if (!match) return 0;
  return Number(match[0].replaceAll(".", "").replace(",", ".")) || 0;
}

function panoramaMax(item: ComparisonColumn, factTypes: string[]) {
  return Math.max(0, ...panoramaValues(item, factTypes).map(panoramaNumber));
}

function panoramaMaturity(item: ComparisonColumn) {
  return Math.max(0, ...panoramaValues(item, ["VADE_SURESI"]).map((text) => {
    const value = panoramaNumber(text);
    return /yıl|yil/i.test(text) ? value * 12 : value;
  }));
}

function panoramaHighestText(item: ComparisonColumn, factTypes: string[], maturity = false) {
  return panoramaValues(item, factTypes).sort((left, right) => {
    const score = (text: string) => maturity && /yıl|yil/i.test(text) ? panoramaNumber(text) * 12 : panoramaNumber(text);
    return score(right) - score(left);
  })[0];
}

function AssistantPanorama({
  data,
  query,
  historical,
  onOpen,
}: {
  data: ComparisonResponse;
  query: string;
  historical: boolean;
  onOpen: (item: ComparisonColumn) => void;
}) {
  const [sortMode, setSortMode] = useState<PanoramaSort>("coverage");
  const baseBanks = useMemo(() => aggregateComparisonItems(data.items), [data.items]);
  const banks = useMemo(() => [...baseBanks].sort((left, right) => {
    if (sortMode === "bank") return left.bank_name.localeCompare(right.bank_name, "tr");
    if (sortMode === "confidence") return (right.confidence ?? 0) - (left.confidence ?? 0);
    if (sortMode === "maturity") return panoramaMaturity(right) - panoramaMaturity(left);
    if (sortMode === "amount") return panoramaMax(right, ["FINANSMAN_TUTARI"]) - panoramaMax(left, ["FINANSMAN_TUTARI"]);
    return comparisonFactCount(right) - comparisonFactCount(left);
  }), [baseBanks, sortMode]);
  const factCount = baseBanks.reduce((total, item) => total + comparisonFactCount(item), 0);
  const coveredBanks = baseBanks.filter((item) => comparisonFactCount(item) > 0).length;
  const mostComplete = [...baseBanks].sort((left, right) => comparisonFactCount(right) - comparisonFactCount(left))[0];
  const longestMaturity = [...baseBanks].sort((left, right) => panoramaMaturity(right) - panoramaMaturity(left))[0];
  const highestAmount = [...baseBanks].sort((left, right) => panoramaMax(right, ["FINANSMAN_TUTARI"]) - panoramaMax(left, ["FINANSMAN_TUTARI"]))[0];
  const highlights = [
    mostComplete && { label: "En geniş bilgi kapsamı", bank: mostComplete, value: `${comparisonFactCount(mostComplete)} bulgu` },
    longestMaturity && panoramaMaturity(longestMaturity) > 0 && { label: "En yüksek görülen vade", bank: longestMaturity, value: panoramaHighestText(longestMaturity, ["VADE_SURESI"], true) },
    highestAmount && panoramaMax(highestAmount, ["FINANSMAN_TUTARI"]) > 0 && { label: "En yüksek görülen tutar", bank: highestAmount, value: panoramaHighestText(highestAmount, ["FINANSMAN_TUTARI"]) },
  ].filter((item): item is { label: string; bank: ComparisonColumn; value: string } => Boolean(item));

  function valueCell(item: ComparisonColumn, factTypes: string[], empty = "Kaynakta yapılandırılmamış") {
    const values = panoramaValues(item, factTypes);
    return <div className={values.length ? panoramaStyles.value : panoramaStyles.emptyValue}>
      <strong>{values[0] ?? empty}</strong>
      {values[1] && <span>{values[1]}</span>}
      {values.length > 2 && <small>+{values.length - 2} ek değer</small>}
    </div>;
  }

  return <section className={panoramaStyles.panel}>
    <header className={panoramaStyles.hero}>
      <div className={panoramaStyles.heroCopy}><span>TÜM BANKALAR · VERİTABANI PANOSU</span><h3>{friendlyCode(data.campaign_type_code, data.campaign_type)}</h3><p>Modelin kısa yanıtından bağımsız olarak ilgili bütün banka kayıtları tek karşılaştırma tablosunda gösterilir.</p></div>
      <div className={panoramaStyles.kpis}><div><b>{banks.length}</b><span>Banka</span></div><div><b>{data.count}</b><span>Kaynak</span></div><div><b>{factCount}</b><span>Bulgu</span></div><div><b>{coveredBanks}/{banks.length}</b><span>Kapsam</span></div></div>
    </header>

    {banks.length ? <>
      {highlights.length > 0 && <div className={panoramaStyles.highlights}>{highlights.map((highlight) => {
        const brand = bankBrand(highlight.bank.bank_name);
        return <article key={highlight.label}><span className={panoramaStyles.highlightMark} style={{ background: brand.color }}>{brand.mark}</span><div><small>{highlight.label}</small><strong>{highlight.value}</strong><span>{shortBank(highlight.bank.bank_name)}</span></div></article>;
      })}</div>}

      <div className={panoramaStyles.toolbar}><div><strong>{formatNumber(banks.length)} kurum listeleniyor</strong><span>{historical ? "Seçili dönemin son güvenli tarihsel kesiti" : "Güncel kabul edilmiş belgeler"}</span></div><label><Icon name="sort" size={16} />Sırala<select onChange={(event) => setSortMode(event.target.value as PanoramaSort)} value={sortMode}><option value="coverage">En fazla bilgi</option><option value="maturity">En yüksek vade</option><option value="amount">En yüksek tutar</option><option value="confidence">En yüksek güven</option><option value="bank">Banka adı</option></select></label></div>

      <div className={panoramaStyles.tableScroll}><div className={panoramaStyles.table} role="table" aria-label="Tüm bankalar finansman karşılaştırması">
        <div className={panoramaStyles.tableHead} role="row"><span>Banka</span><span>Finansman tutarı</span><span>Vade</span><span>Kâr payı / oran</span><span>Masraf ve ücretler</span><span>Kaynak</span></div>
        {banks.map((item) => {
          const brand = bankBrand(item.bank_name);
          return <div className={panoramaStyles.row} role="row" key={item.bank_name}>
            <div className={panoramaStyles.bankCell} role="cell"><span className={panoramaStyles.brandMark} style={{ background: brand.color }}>{brand.mark}</span><div><strong>{shortBank(item.bank_name)}</strong><span>{item.document_count} belge · {comparisonFactCount(item)} bulgu</span><small>{scoreLabel(item.confidence)} sınıflandırma güveni</small></div></div>
            <div role="cell" data-label="Finansman tutarı">{valueCell(item, ["FINANSMAN_TUTARI"])}</div>
            <div role="cell" data-label="Vade">{valueCell(item, ["VADE_SURESI"])}</div>
            <div role="cell" data-label="Kâr payı / oran">{valueCell(item, ["KAR_PAYI_ORANI", "FINANSMAN_ORANI"])}</div>
            <div role="cell" data-label="Masraf ve ücretler">{valueCell(item, ["TAHSIS_UCRETI", "EKSPERTIZ_UCRETI", "IPOTEK_TESIS_UCRETI", "DIGER_UCRET", "MASRAF_DURUMU"])}</div>
            <div className={panoramaStyles.sourceCell} role="cell"><button onClick={() => onOpen(item)} type="button">İncele <Icon name="arrow" size={16} /></button><span>{item.page_titles.length} başlık</span></div>
          </div>;
        })}
      </div></div>

      <details className={panoramaStyles.textSummary}><summary><Icon name="document" size={15} /> Tüm bankaları metin özeti olarak göster</summary><div>{banks.map((item) => {
        const facts = prioritizedFacts(query, data.campaign_type_code, item).slice(0, 4);
        return <p key={item.bank_name}><b>{shortBank(item.bank_name)}:</b> {facts.length ? facts.map(({ factType, values }) => `${factLabels[factType] ?? friendlyCode(factType)} ${values[0]?.text ?? "—"}`).join("; ") : "İlgili belge var; yapılandırılmış karşılaştırma alanı yok."}</p>;
      })}</div></details>
    </> : <div className={panoramaStyles.noData}><Icon name="warning" size={20} /><span>Seçilen ürün için banka bazlı kayıt bulunamadı.</span></div>}
    <footer className={panoramaStyles.note}><Icon name="shield" size={15} /><span>Boş hücre, bankanın ürünü sunmadığı anlamına gelmez; yalnızca kabul edilmiş kaynaklarda o alanın yapılandırılmadığını gösterir.</span></footer>
  </section>;
}

export default function Home() {
  const [view, setView] = useState<View>("overview");
  const [overviewScope, setOverviewScope] = useState<OverviewScope>("all");
  const [mobileMenu, setMobileMenu] = useState(false);
  const [connection, setConnection] = useState<ConnectionState>("checking");
  const [demoMode, setDemoMode] = useState(false);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [dashboard, setDashboard] = useState<DashboardOverview>(demoDashboard);
  const [historyOverview, setHistoryOverview] = useState<HistoricalOverview>(demoHistory);
  const [options, setOptions] = useState<ComparisonOptions>(demoOptions);
  const [catalog, setCatalog] = useState<CatalogResponse>(demoCatalog);
  const [catalogScope, setCatalogScope] = useState<CatalogScope>("all");
  const [catalogPeriod, setCatalogPeriod] = useState<HistoryPeriod>("1y");
  const [historicalResults, setHistoricalResults] = useState<HistoricalSearchResult[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [catalogLoading, setCatalogLoading] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [productFilter, setProductFilter] = useState("");
  const [bankFilter, setBankFilter] = useState("");
  const [factFilter, setFactFilter] = useState("all");
  const [minConfidence, setMinConfidence] = useState("0");
  const [sortBy, setSortBy] = useState("relevance");
  const [sortOrder, setSortOrder] = useState("desc");
  const [detail, setDetail] = useState<DocumentDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [selectedProduct, setSelectedProduct] = useState("KONUT_FINANSMANI");
  const [selectedBanks, setSelectedBanks] = useState<string[]>([]);
  const [compareScope, setCompareScope] = useState<CompareScope>("live");
  const [comparePeriod, setComparePeriod] = useState<HistoryPeriod>("3m");
  const [comparison, setComparison] = useState<ComparisonResponse | null>(null);
  const [comparisonBaseline, setComparisonBaseline] = useState<ComparisonResponse | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareSort, setCompareSort] = useState("confidence");
  const [comparisonView, setComparisonView] = useState<ComparisonView>("cards");
  const [chatInput, setChatInput] = useState("");
  const [assistantScope, setAssistantScope] = useState<CompareScope>("live");
  const [assistantPeriod, setAssistantPeriod] = useState<HistoryPeriod>("1y");
  const [assistantProduct, setAssistantProduct] = useState("auto");
  const [conversation, setConversation] = useState<ConversationItem[]>([]);
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState<string | null>(null);
  const chatAbort = useRef<AbortController | null>(null);
  const messageId = useRef(1);

  useEffect(() => {
    let cancelled = false;
    async function boot() {
      try {
        const [healthData, overviewData, historyData, optionsData, catalogData] = await Promise.all([
          apiRequest<HealthResponse>("/health", {}, 15000),
          apiRequest<DashboardOverview>("/dashboard/overview", {}, 15000),
          apiRequest<HistoricalOverview>("/history/overview", {}, 15000),
          apiRequest<ComparisonOptions>("/comparison/options", {}, 15000),
          apiRequest<CatalogResponse>("/catalog/search", {
            method: "POST",
            body: JSON.stringify({ page: 1, page_size: 12, sort_by: "confidence", sort_order: "desc" }),
          }, 15000),
        ]);
        if (cancelled) return;
        setHealth(healthData);
        setDashboard(overviewData);
        setHistoryOverview(historyData);
        const historicalProducts = historyData.product_types
          .filter((item) => item.code && !optionsData.campaign_types.some((option) => option.code === item.code))
          .map((item) => ({ code: item.code!, label: friendlyCode(item.code!), document_count: item.count, bank_count: 0 }));
        const historicalBanks = historyData.banks.map((item) => item.name).filter((item): item is string => Boolean(item));
        const mergedOptions = {
          ...optionsData,
          campaign_types: [...optionsData.campaign_types, ...historicalProducts],
          banks: [...new Set([...optionsData.banks, ...historicalBanks])],
        };
        setOptions(mergedOptions);
        setCatalog(catalogData);
        setSelectedBanks([]);
        setSelectedProduct(
          mergedOptions.campaign_types.some((item) => item.code === "KONUT_FINANSMANI")
            ? "KONUT_FINANSMANI"
            : mergedOptions.campaign_types[0]?.code ?? "KONUT_FINANSMANI",
        );
        setDemoMode(false);
        setConnection(healthData.ollama_model_ready === false ? "degraded" : "online");
      } catch (error) {
        if (cancelled) return;
        console.error(error);
        setDashboard(demoDashboard);
        setHistoryOverview(demoHistory);
        setOptions(demoOptions);
        setCatalog(demoCatalog);
        setSelectedBanks([]);
        setDemoMode(true);
        setConnection("offline");
      }
    }
    boot();
    return () => {
      cancelled = true;
      chatAbort.current?.abort();
    };
  }, []);

  const liveDocumentCount = dashboard.live_document_count ?? dashboard.document_count;
  const historicalDocumentCount = historyOverview.historical_document_count;
  const totalInventoryCount = dashboard.total_snapshot_count ?? liveDocumentCount + historicalDocumentCount;
  const totalFactCount = dashboard.fact_count + historyOverview.historical_fact_count;
  const totalChunkCount = (health?.chunk_count ?? 1772) + historyOverview.historical_chunk_count;
  const historicalCoverage = historicalDocumentCount
    ? Math.round((historyOverview.searchable_document_count / historicalDocumentCount) * 1000) / 10
    : 0;
  const archiveProducts = useMemo(
    () => historicalBuckets(historyOverview.product_types, "product"),
    [historyOverview.product_types],
  );
  const archiveBanks = useMemo(
    () => historicalBuckets(historyOverview.banks, "bank"),
    [historyOverview.banks],
  );
  const visibleProducts = useMemo(() => {
    if (overviewScope === "live") return dashboard.product_types;
    if (overviewScope === "history") return archiveProducts;
    return mergeDashboardBuckets(dashboard.product_types, archiveProducts, "product");
  }, [archiveProducts, dashboard.product_types, overviewScope]);
  const visibleBanks = useMemo(() => {
    if (overviewScope === "live") return dashboard.banks;
    if (overviewScope === "history") return archiveBanks;
    return mergeDashboardBuckets(dashboard.banks, archiveBanks, "bank");
  }, [archiveBanks, dashboard.banks, overviewScope]);
  const catalogDateRange = useMemo(
    () => periodDates(catalogPeriod, historyOverview),
    [catalogPeriod, historyOverview],
  );
  const activePeriodLabel = historyPeriodOptions.find((item) => item.value === comparePeriod)?.label ?? "Tüm arşiv";
  const assistantQuestions = assistantScope === "history" ? historicalQuickQuestions : quickQuestions;
  const assistantPeriodLabel = historyPeriodOptions.find((item) => item.value === assistantPeriod)?.label ?? "Tüm arşiv";

  const navItems: { id: View; label: string; icon: IconName; badge?: string }[] = [
    { id: "overview", label: "Genel bakış", icon: "home" },
    { id: "catalog", label: "Ürün kataloğu", icon: "search", badge: formatNumber(totalInventoryCount) },
    { id: "compare", label: "Karşılaştırma", icon: "compare" },
    { id: "assistant", label: "Akıllı asistan", icon: "spark" },
    { id: "quality", label: "Veri kalitesi", icon: "chart" },
  ];

  const viewMeta: Record<View, { eyebrow: string; title: string; description: string }> = {
    overview: { eyebrow: "KARAR MERKEZİ", title: "Katılım finansı panoraması", description: "Veri setinin kapsamını, dağılımını ve bilgi yoğunluğunu gerçek zamanlı izleyin." },
    catalog: { eyebrow: "AKILLI KATALOG", title: "Ürün ve belge keşfi", description: "Banka, ürün, güven ve bilgi kapsamına göre bütün kayıtları filtreleyin." },
    compare: { eyebrow: "KARŞILAŞTIRMA LABORATUVARI", title: "Koşulları yan yana görün", description: "Çıkarılmış finansman bilgilerini kanıtlarıyla birlikte karşılaştırın." },
    assistant: { eyebrow: "KAYNAKLI YAPAY ZEKÂ", title: "HititFinLex Asistan", description: "Hibrit arama ile bulunan resmî kaynaklardan doğrudan yanıt alın." },
    quality: { eyebrow: "VERİ YÖNETİŞİMİ", title: "Kalite ve kapsama görünümü", description: "Sınıflandırma güvenini, çıkarılmış alanları ve inceleme kuyruğunu takip edin." },
  };

  function changeView(next: View) {
    setView(next);
    setMobileMenu(false);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  async function loadCatalog(
    page = 1,
    queryOverride?: string,
    overrides?: {
      productFilter?: string;
      bankFilter?: string;
      factFilter?: string;
      minConfidence?: string;
      sortBy?: string;
      sortOrder?: string;
      catalogScope?: CatalogScope;
      catalogPeriod?: HistoryPeriod;
    },
  ) {
    const effectiveQuery = queryOverride ?? query;
    const effectiveProduct = overrides?.productFilter ?? productFilter;
    const effectiveBank = overrides?.bankFilter ?? bankFilter;
    const effectiveFact = overrides?.factFilter ?? factFilter;
    const effectiveConfidence = overrides?.minConfidence ?? minConfidence;
    const effectiveSort = overrides?.sortBy ?? sortBy;
    const effectiveOrder = overrides?.sortOrder ?? sortOrder;
    const effectiveScope = overrides?.catalogScope ?? catalogScope;
    const effectivePeriod = overrides?.catalogPeriod ?? catalogPeriod;
    const effectiveDates = periodDates(effectivePeriod, historyOverview);
    setCatalogLoading(true);
    setHistoryLoading(effectiveScope !== "live");
    setCatalogError(null);
    try {
      const liveRequest = effectiveScope === "history"
        ? Promise.resolve<CatalogResponse | null>(null)
        : apiRequest<CatalogResponse>("/catalog/search", {
            method: "POST",
            body: JSON.stringify({
              query: effectiveQuery,
              product_types: effectiveProduct ? [effectiveProduct] : [],
              bank_names: effectiveBank ? [effectiveBank] : [],
              has_facts: effectiveFact === "with" ? true : effectiveFact === "without" ? false : null,
              min_confidence: Number(effectiveConfidence),
              sort_by: effectiveSort,
              sort_order: effectiveOrder,
              page,
              page_size: 12,
            }),
          });
      const historyRequest = effectiveScope === "live" || effectiveQuery.trim().length < 2
        ? Promise.resolve<HistoricalSearchResponse | null>(null)
        : apiRequest<HistoricalSearchResponse>("/history/search", {
            method: "POST",
            body: JSON.stringify({
              query: effectiveQuery,
              top_k: 12,
              product_types: effectiveProduct ? [effectiveProduct] : [],
              bank_names: effectiveBank ? [effectiveBank] : [],
              date_from: effectiveDates.dateFrom,
              date_to: effectiveDates.dateTo,
            }),
          }, 60000);
      const [liveResult, historyResult] = await Promise.all([liveRequest, historyRequest]);
      if (liveResult) setCatalog(liveResult);
      if (effectiveScope === "history" && !liveResult) {
        setCatalog({ total: 0, page: 1, page_size: 12, page_count: 0, items: [] });
      }
      setHistoricalResults(historyResult?.results ?? []);
      setDemoMode(false);
      setConnection((current) => current === "degraded" ? current : "online");
    } catch (error) {
      console.error(error);
      const normalizedQuery = effectiveQuery.toLocaleLowerCase("tr-TR");
      let items = demoCatalog.items.filter((item) => {
        const matchesQuery = !normalizedQuery || `${item.page_title} ${item.bank_name} ${item.campaign_type_code}`.toLocaleLowerCase("tr-TR").includes(normalizedQuery);
        const matchesProduct = !effectiveProduct || item.campaign_type_code === effectiveProduct;
        const matchesBank = !effectiveBank || item.bank_name === effectiveBank;
        const matchesConfidence = (item.confidence ?? 0) >= Number(effectiveConfidence);
        const matchesFacts = effectiveFact === "with" ? item.fact_count > 0 : effectiveFact === "without" ? item.fact_count === 0 : true;
        return matchesQuery && matchesProduct && matchesBank && matchesConfidence && matchesFacts;
      });
      items = [...items].sort((a, b) => {
        const direction = effectiveOrder === "asc" ? 1 : -1;
        if (effectiveSort === "bank") return a.bank_name.localeCompare(b.bank_name, "tr") * direction;
        if (effectiveSort === "title") return (a.page_title ?? "").localeCompare(b.page_title ?? "", "tr") * direction;
        if (effectiveSort === "facts") return (a.fact_count - b.fact_count) * direction;
        return ((a.confidence ?? 0) - (b.confidence ?? 0)) * direction;
      });
      if (effectiveScope !== "history") {
        setCatalog({ total: items.length, page: 1, page_size: 12, page_count: items.length ? 1 : 0, items });
      }
      setHistoricalResults([]);
      setCatalogError("Canlı katalog yerine veri seti önizlemesi filtreleniyor. Tam sonuçlar için API V2.5’i çalıştırın.");
    } finally {
      setCatalogLoading(false);
      setHistoryLoading(false);
    }
  }

  function submitGlobalSearch(event: FormEvent) {
    event.preventDefault();
    changeView("catalog");
    loadCatalog(1);
  }

  async function openDetail(documentId: number) {
    if (demoMode) {
      const item = catalog.items.find((candidate) => candidate.document_id === documentId);
      if (!item) return;
      setDetail({
        ...item,
        raw_text: "Yerel API bağlı olmadığından ham belge gösterilemiyor.",
        facts: [],
      });
      return;
    }
    setDetailLoading(true);
    try {
      setDetail(await apiRequest<DocumentDetail>(`/documents/${documentId}`));
    } catch (error) {
      console.error(error);
      setCatalogError("Belge ayrıntısı yüklenemedi.");
    } finally {
      setDetailLoading(false);
    }
  }

  function toggleBank(bank: string) {
    setSelectedBanks((current) => {
      if (current.includes(bank)) return current.filter((item) => item !== bank);
      return [...current, bank];
    });
  }

  async function runComparison() {
    setCompareLoading(true);
    try {
      if (compareScope === "live") {
        const data = await apiRequest<ComparisonResponse>("/comparison", {
          method: "POST",
          body: JSON.stringify({
            campaign_type_code: selectedProduct,
            bank_names: selectedBanks,
            limit: 100,
          }),
        });
        setComparison(data);
        setComparisonBaseline(null);
      } else {
        const dates = periodDates(comparePeriod, historyOverview);
        const request = (asOf: string | null) => apiRequest<HistoricalComparisonResponse>("/history/comparison", {
          method: "POST",
          body: JSON.stringify({
            product_type_code: selectedProduct,
            bank_names: selectedBanks,
            as_of: asOf,
            limit: 50,
          }),
        }, 60000);
        const [currentData, baselineData] = await Promise.all([
          request(dates.dateTo),
          dates.dateFrom && comparePeriod !== "all" ? request(dates.dateFrom) : Promise.resolve(null),
        ]);
        setComparison(normalizeHistoricalComparison(currentData));
        setComparisonBaseline(baselineData ? normalizeHistoricalComparison(baselineData) : null);
      }
      setDemoMode(false);
    } catch (error) {
      console.error(error);
      const fallback = selectedProduct === "KONUT_FINANSMANI"
        ? {
            ...demoComparison,
            items: demoComparison.items.filter((item) =>
              !selectedBanks.length || selectedBanks.some((bank) => shortBank(bank) === shortBank(item.bank_name)),
            ),
          }
        : { campaign_type_code: selectedProduct, campaign_type: null, count: 0, items: [] };
      setComparison({ ...fallback, count: fallback.items.length });
      setComparisonBaseline(null);
    } finally {
      setCompareLoading(false);
    }
  }

  const sortedComparison = useMemo(() => {
    if (!comparison) return [];
    const columns = aggregateComparisonItems(comparison.items);
    if (compareSort === "facts") return columns.sort((a, b) => comparisonFactCount(b) - comparisonFactCount(a));
    if (compareSort === "bank") return columns.sort((a, b) => a.bank_name.localeCompare(b.bank_name, "tr"));
    return columns.sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  }, [comparison, compareSort]);

  const activeMatrixFacts = useMemo(
    () => visibleComparisonFacts(comparison?.campaign_type_code ?? selectedProduct, sortedComparison),
    [comparison?.campaign_type_code, selectedProduct, sortedComparison],
  );

  const comparisonTrend = useMemo(() => {
    if (compareScope !== "history" || !comparisonBaseline) return null;
    const current = aggregateComparisonItems(comparison?.items ?? []);
    const baseline = aggregateComparisonItems(comparisonBaseline.items);
    const signature = (item: ComparisonColumn) => JSON.stringify(
      Object.entries(item.attributes)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([fact, values]) => [fact, values.map((value) => value.text).sort()]),
    );
    const currentMap = new Map(current.map((item) => [item.bank_name, item]));
    const baselineMap = new Map(baseline.map((item) => [item.bank_name, item]));
    const banks = new Set([...currentMap.keys(), ...baselineMap.keys()]);
    const changedBanks = [...banks].filter((bank) => {
      const latest = currentMap.get(bank);
      const earlier = baselineMap.get(bank);
      return !latest || !earlier || signature(latest) !== signature(earlier);
    }).length;
    const currentFacts = current.reduce((sum, item) => sum + comparisonFactCount(item), 0);
    const baselineFacts = baseline.reduce((sum, item) => sum + comparisonFactCount(item), 0);
    return { changedBanks, factDelta: currentFacts - baselineFacts, baselineBanks: baseline.length };
  }, [compareScope, comparison, comparisonBaseline]);

  function openComparisonItem(item: ComparisonColumn) {
    if (compareScope === "history") {
      if (item.source_url) window.open(item.source_url, "_blank", "noopener,noreferrer");
      return;
    }
    openDetail(item.document_id);
  }

  async function submitChat(question: string) {
    const clean = question.trim();
    if (!clean || chatLoading) return;
    setChatInput("");
    setChatError(null);
    setChatLoading(true);
    const controller = new AbortController();
    chatAbort.current = controller;
    const timer = window.setTimeout(() => controller.abort(), 180000);
    try {
      const dates = periodDates(assistantPeriod, historyOverview);
      const productCode = assistantProduct === "auto" ? inferProductType(clean) : assistantProduct;
      const chatPromise = (async () => {
        const response = await fetch(`${API_BASE_URL}${assistantScope === "history" ? "/history/chat" : "/chat"}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(assistantScope === "history"
            ? { query: clean, top_k: 8, date_from: dates.dateFrom, date_to: dates.dateTo }
            : { query: clean, top_k: 8 }),
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(await response.text());
        return response.json();
      })();
      const panoramaPromise = productCode ? (async () => {
        const response = await fetch(`${API_BASE_URL}${assistantScope === "history" ? "/history/comparison" : "/comparison"}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(assistantScope === "history"
            ? { product_type_code: productCode, bank_names: [], as_of: dates.dateTo, limit: 50 }
            : { campaign_type_code: productCode, bank_names: [], limit: 100 }),
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(await response.text());
        const payload = await response.json();
        return assistantScope === "history"
          ? normalizeHistoricalComparison(payload as HistoricalComparisonResponse)
          : payload as ComparisonResponse;
      })() : Promise.resolve<ComparisonResponse | null>(null);

      const [chatResult, panoramaResult] = await Promise.allSettled([chatPromise, panoramaPromise]);
      if (chatResult.status === "rejected") throw chatResult.reason;
      const panorama = panoramaResult.status === "fulfilled" ? panoramaResult.value : null;
      let normalized: ChatResponse;
      if (assistantScope === "history") {
        const data = chatResult.value as HistoricalChatResponse;
        normalized = {
          query: data.query,
          answer: data.answer,
          model: data.model,
          sources: data.sources.map((source) => ({ ...source, source_id: source.rank })),
        };
      } else {
        normalized = chatResult.value as ChatResponse;
      }
      setConversation((current) => [...current, {
        ...normalized,
        id: messageId.current++,
        scope: assistantScope,
        periodLabel: assistantScope === "history"
          ? historyPeriodOptions.find((item) => item.value === assistantPeriod)?.label
          : undefined,
        panorama,
        panoramaProduct: productCode,
      }]);
      setConnection("online");
    } catch (error) {
      const aborted = (error as Error).name === "AbortError";
      setChatError(aborted ? "İstek durduruldu veya zaman aşımına uğradı." : "Asistan yanıt veremedi. API, Ollama ve CORS durumunu kontrol edin.");
    } finally {
      window.clearTimeout(timer);
      chatAbort.current = null;
      setChatLoading(false);
    }
  }

  function stopChat() {
    chatAbort.current?.abort();
  }

  function resetFilters() {
    setQuery("");
    setProductFilter("");
    setBankFilter("");
    setFactFilter("all");
    setMinConfidence("0");
    setSortBy("relevance");
    setSortOrder("desc");
    loadCatalog(1, "", {
      productFilter: "",
      bankFilter: "",
      factFilter: "all",
      minConfidence: "0",
      sortBy: "relevance",
      sortOrder: "desc",
    });
  }

  return (
    <main className="workspace-shell">
      <aside className={mobileMenu ? "side-rail open" : "side-rail"}>
        <div className="rail-brand">
          <span className="brand-mark">H</span>
          <div><strong>HititFinLex</strong><small>Katılım Finans Karar Platformu</small></div>
          <button className="mobile-close" onClick={() => setMobileMenu(false)} type="button"><Icon name="close" /></button>
        </div>
        <nav className="rail-nav">
          <small>ÇALIŞMA ALANI</small>
          {navItems.map((item) => (
            <button className={view === item.id ? "active" : ""} key={item.id} onClick={() => changeView(item.id)} type="button">
              <Icon name={item.icon} size={19} /><span>{item.label}</span>{item.badge && <b>{item.badge}</b>}
            </button>
          ))}
        </nav>
        <div className="rail-system">
          <div className={`system-orb ${connection}`}><Icon name="database" size={19} /></div>
          <div><strong>{connection === "online" ? "Tüm sistemler hazır" : connection === "degraded" ? "Veri hazır, LLM bekleniyor" : connection === "checking" ? "Sistem kontrol ediliyor" : "Yerel API bekleniyor"}</strong><small>{health?.gpu ?? "PostgreSQL + BGE-M3 + Qwen"}</small></div>
        </div>
        <div className="rail-footer"><Icon name="shield" size={15} /><span>Yerel • Kaynaklı • Denetlenebilir</span></div>
      </aside>

      {mobileMenu && <button aria-label="Menüyü kapat" className="rail-backdrop" onClick={() => setMobileMenu(false)} type="button" />}

      <section className="workspace-main">
        <header className="workspace-topbar">
          <button className="menu-button" onClick={() => setMobileMenu(true)} type="button"><Icon name="menu" /></button>
          <form className="global-search" onSubmit={submitGlobalSearch}>
            <Icon name="search" size={18} />
            <input aria-label="Tüm belgelerde ara" onChange={(event) => setQuery(event.target.value)} placeholder={`${formatNumber(totalInventoryCount)} canlı ve tarihsel kayıtta ara…`} value={query} />
            <kbd>ENTER</kbd>
          </form>
          <div className="topbar-actions">
            {demoMode && <span className="demo-chip">VERİ ÖNİZLEME</span>}
            <span className={`connection-chip ${connection}`}><i />{connection === "online" ? "Canlı veri" : connection === "degraded" ? "Kısmi hazır" : connection === "checking" ? "Bağlanıyor" : "API çevrimdışı"}</span>
          </div>
        </header>

        <div className="workspace-content">
          {demoMode && <div className="api-banner"><Icon name="warning" size={18} /><div><strong>Dashboard örnek veri görünümünde.</strong><span>Gerçek filtreleme ve belge ayrıntıları için HititFinLex API V2.5’i çalıştırın.</span></div><button onClick={() => window.location.reload()} type="button"><Icon name="refresh" size={15} /> Yeniden dene</button></div>}

          <div className="page-heading">
            <div><span>{viewMeta[view].eyebrow}</span><h1>{viewMeta[view].title}</h1><p>{viewMeta[view].description}</p></div>
            <div className="page-heading-actions">
              <button className="ghost-button" onClick={() => changeView("assistant")} type="button"><Icon name="spark" size={17} /> Asistana sor</button>
              <button className="primary-button" onClick={() => changeView("catalog")} type="button"><Icon name="search" size={17} /> Kataloğu aç</button>
            </div>
          </div>

          {view === "overview" && (
            <div className="view-stack">
              <section className="scope-toolbar panel" aria-label="Veri kapsamı">
                <div><span><Icon name="database" size={16} /></span><div><strong>Veri evreni</strong><small>Dağılımları canlı envanter, tarihsel arşiv veya birleşik görünümde inceleyin.</small></div></div>
                <div className="segmented-control">
                  <button className={overviewScope === "all" ? "active" : ""} onClick={() => setOverviewScope("all")} type="button">Birleşik <b>{formatNumber(totalInventoryCount)}</b></button>
                  <button className={overviewScope === "live" ? "active" : ""} onClick={() => setOverviewScope("live")} type="button">Güncel <b>{formatNumber(liveDocumentCount)}</b></button>
                  <button className={overviewScope === "history" ? "active" : ""} onClick={() => setOverviewScope("history")} type="button">Arşiv <b>{formatNumber(historicalDocumentCount)}</b></button>
                </div>
              </section>
              <section className="kpi-grid">
                <article className="kpi-card accent"><div className="kpi-icon"><Icon name="document" /></div><div><span>Toplam veri envanteri</span><strong>{formatNumber(totalInventoryCount)}</strong><small>{formatNumber(liveDocumentCount)} güncel · {formatNumber(historicalDocumentCount)} tarihsel</small></div><b>CANLI + ARŞİV</b></article>
                <article className="kpi-card"><div className="kpi-icon blue"><Icon name="layers" /></div><div><span>Çıkarılmış bilgi</span><strong>{formatNumber(totalFactCount)}</strong><small>{formatNumber(totalChunkCount)} semantik arama parçası</small></div></article>
                <article className="kpi-card"><div className="kpi-icon gold"><Icon name="chart" /></div><div><span>Canlı bilgi kapsaması</span><strong>%{dashboard.coverage_percentage.toLocaleString("tr-TR")}</strong><small>{formatNumber(dashboard.documents_with_facts)} güncel belgede yapılandırılmış alan</small></div></article>
                <article className="kpi-card"><div className="kpi-icon violet"><Icon name="shield" /></div><div><span>Ortalama güven</span><strong>%{Math.round(dashboard.average_confidence * 100)}</strong><small>{dashboard.pending_document_reviews + dashboard.pending_fact_reviews} bekleyen inceleme</small></div></article>
              </section>

              <section className="archive-ribbon">
                <div className="archive-lead"><span><Icon name="clock" size={21} /></span><div><small>TARİHSEL ZEKÂ AKTİF</small><h2>{formatDate(historyOverview.history_start_date)} — {formatDate(historyOverview.history_end_date)}</h2><p>Katılım finansının yaklaşık on yıllık değişimini kaynak, tarih ve ürün türüyle izleyin.</p></div></div>
                <div className="archive-metrics"><div><span>Aranabilir arşiv</span><strong>{formatNumber(historyOverview.searchable_document_count)}</strong><small>%{historicalCoverage} güvenli kapsam</small></div><div><span>Tarihsel bilgi</span><strong>{formatNumber(historyOverview.historical_fact_count)}</strong><small>kabul edilmiş alan</small></div><div><span>Vektör parçası</span><strong>{formatNumber(historyOverview.embedded_chunk_count)}</strong><small>eksiksiz embedding</small></div><div><span>Kaynak kurum</span><strong>{historyOverview.banks.length}</strong><small>katılım bankası</small></div></div>
                <button onClick={() => { setCatalogScope("history"); setView("catalog"); }} type="button">Arşivde keşfe çık <Icon name="arrow" size={16} /></button>
              </section>

              <section className="dashboard-grid">
                <article className="panel product-panel">
                  <div className="panel-heading"><div><span>ÜRÜN DAĞILIMI</span><h2>Veri setinin ürün haritası</h2></div><button onClick={() => changeView("catalog")} type="button">Tüm kayıtlar <Icon name="arrow" size={15} /></button></div>
                  <div className="product-bars">
                    {visibleProducts.slice(0, 8).map((item, index) => (
                      <button key={item.code} onClick={() => { const nextScope = overviewScope === "live" ? "live" : overviewScope === "history" ? "history" : "all"; setProductFilter(item.code); setCatalogScope(nextScope); changeView("catalog"); loadCatalog(1, "", { productFilter: item.code, catalogScope: nextScope }); }} type="button">
                        <span className={`bar-index tone-${index % 6}`}>{String(index + 1).padStart(2, "0")}</span>
                        <div><strong>{friendlyCode(item.code, item.label)}</strong><i><u style={{ width: `${Math.max(item.percentage, 3)}%` }} /></i></div>
                        <b>{formatNumber(item.count)}</b><small>%{item.percentage}</small>
                      </button>
                    ))}
                  </div>
                </article>

                <article className="panel bank-panel">
                  <div className="panel-heading"><div><span>KURUM KAPSAMI</span><h2>Bankalara göre belgeler</h2></div><div className="live-mark"><i /> {overviewScope === "all" ? "Birleşik" : overviewScope === "history" ? "Arşiv" : "Güncel"}</div></div>
                  <div className="bank-chart">
                    {visibleBanks.slice(0, 10).map((bank) => (
                      <button key={bank.code} onClick={() => { const nextScope = overviewScope === "live" ? "live" : overviewScope === "history" ? "history" : "all"; setBankFilter(bank.label); setCatalogScope(nextScope); changeView("catalog"); loadCatalog(1, "", { bankFilter: bank.label, catalogScope: nextScope }); }} type="button">
                        <BankLogo name={bank.label} />
                        <div><strong>{shortBank(bank.label)}</strong><i><u style={{ width: `${Math.max(bank.percentage * 6.7, 4)}%` }} /></i></div>
                        <b>{bank.count}</b>
                      </button>
                    ))}
                  </div>
                </article>
              </section>

              <section className="dashboard-grid lower">
                <article className="panel facts-panel">
                  <div className="panel-heading"><div><span>BİLGİ ÇIKARIMI</span><h2>En yoğun alanlar</h2></div><span className="soft-count">{formatNumber(dashboard.fact_count)} toplam</span></div>
                  <div className="fact-cloud">
                    {dashboard.fact_types.map((fact, index) => <div className={`fact-stat fact-${index % 5}`} key={fact.code}><span>{fact.label}</span><strong>{formatNumber(fact.count)}</strong><small>tüm bilgilerin %{fact.percentage}’i</small></div>)}
                  </div>
                </article>

                <article className="panel action-panel">
                  <span className="action-symbol"><Icon name="spark" size={26} /></span><span>AKILLI KEŞİF</span><h2>Veriden doğrudan<br/>karara geçin.</h2><p>Doğal dilde sorun, ilgili ürünleri hibrit aramayla bulun ve resmî kanıtları tek ekranda inceleyin.</p><button onClick={() => { changeView("assistant"); setChatInput("Konut finansmanı ürünlerini vade ve masraflarıyla karşılaştır"); }} type="button">Yeni analiz başlat <Icon name="arrow" size={17} /></button>
                </article>
              </section>

              <section className="panel recent-panel">
                <div className="panel-heading"><div><span>SON HAREKETLER</span><h2>Güncellenen belgeler</h2></div><button onClick={() => changeView("catalog")} type="button">Kataloğa git <Icon name="arrow" size={15} /></button></div>
                {dashboard.latest_documents.length ? <div className="data-table recent-table"><div className="table-head"><span>Banka / belge</span><span>Ürün türü</span><span>Güven</span><span>Güncelleme</span><span /></div>{dashboard.latest_documents.map((item) => <button className="table-row" key={item.document_id} onClick={() => openDetail(item.document_id)} type="button"><span className="title-cell"><BankLogo name={item.bank_name} /><span><strong>{item.page_title ?? "Başlıksız belge"}</strong><small>{shortBank(item.bank_name)}</small></span></span><span><em className="type-pill">{friendlyCode(item.campaign_type_code, item.campaign_type)}</em></span><span><b className="confidence-value">{scoreLabel(item.confidence)}</b></span><span>{formatDate(item.updated_at)}</span><span><Icon name="chevron" size={15} /></span></button>)}</div> : <div className="empty-inline"><Icon name="clock" /><div><strong>Son güncellemeler canlı API ile görünür.</strong><span>API V2.5 bağlandığında belge hareketleri burada listelenecek.</span></div></div>}
              </section>
            </div>
          )}

          {view === "catalog" && (
            <div className="view-stack">
              <section className="catalog-scope-bar panel">
                <div className="scope-copy"><span><Icon name="database" size={17} /></span><div><strong>Arama evreni</strong><small>Canlı ürünleri ve tarihsel sayfa kesitlerini aynı sorguda tarayın.</small></div></div>
                <div className="segmented-control compact">
                  {(["all", "live", "history"] as CatalogScope[]).map((scope) => <button className={catalogScope === scope ? "active" : ""} key={scope} onClick={() => { setCatalogScope(scope); loadCatalog(1, undefined, { catalogScope: scope }); }} type="button">{scope === "all" ? "Tümü" : scope === "live" ? "Güncel" : "Tarihsel"}</button>)}
                </div>
                {/* Guncel kapsaminda donem secimi anlamsiz; cipleri gizlerken yerlerini koruruz ki ustteki kontrol yerinden oynamasin. */}
                <div className={catalogScope === "live" ? "period-chips hidden" : "period-chips"} aria-label="Tarih aralığı">{historyPeriodOptions.map((period) => <button className={catalogPeriod === period.value ? "active" : ""} key={period.value} onClick={() => { setCatalogPeriod(period.value); loadCatalog(1, undefined, { catalogPeriod: period.value }); }} type="button">{period.label}</button>)}</div>
              </section>
              <section className="filter-panel panel">
                <div className="filter-top"><div className="catalog-search"><Icon name="search" size={18} /><input onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") loadCatalog(1); }} placeholder="Ürün adı, koşul veya belge metni…" value={query} /><button onClick={() => loadCatalog(1)} type="button">Ara</button></div><button className="reset-button" onClick={resetFilters} type="button"><Icon name="refresh" size={15} /> Filtreleri temizle</button></div>
                <div className="filter-grid">
                  <label><span><Icon name="layers" size={14} /> Ürün türü</span><select onChange={(event) => setProductFilter(event.target.value)} value={productFilter}><option value="">Tüm ürünler</option>{options.campaign_types.map((item) => <option key={item.code} value={item.code}>{friendlyCode(item.code, item.label)} ({item.document_count})</option>)}</select></label>
                  <label><span><Icon name="building" size={14} /> Banka</span><select onChange={(event) => setBankFilter(event.target.value)} value={bankFilter}><option value="">Tüm bankalar</option>{options.banks.map((bank) => <option key={bank} value={bank}>{shortBank(bank)}</option>)}</select></label>
                  <label><span><Icon name="database" size={14} /> Bilgi kapsamı</span><select onChange={(event) => setFactFilter(event.target.value)} value={factFilter}><option value="all">Tüm kayıtlar</option><option value="with">Çıkarılmış bilgisi olan</option><option value="without">Bilgisi eksik olan</option></select></label>
                  <label><span><Icon name="shield" size={14} /> En az güven</span><select onChange={(event) => setMinConfidence(event.target.value)} value={minConfidence}><option value="0">Tüm güven düzeyleri</option><option value="0.7">%70 ve üzeri</option><option value="0.8">%80 ve üzeri</option><option value="0.9">%90 ve üzeri</option><option value="0.95">%95 ve üzeri</option></select></label>
                </div>
                <div className="filter-bottom"><span><Icon name="filter" size={14} /> {formatNumber((catalogScope === "history" ? 0 : catalog.total) + historicalResults.length)} sonuç gösteriliyor</span><div><label>Sırala<select disabled={catalogScope === "history"} onChange={(event) => setSortBy(event.target.value)} value={sortBy}><option value="relevance">En alakalı</option><option value="confidence">Güven puanı</option><option value="facts">Bilgi yoğunluğu</option><option value="updated">En güncel</option><option value="bank">Banka adı</option><option value="title">Ürün adı</option></select></label><button aria-label="Sıralama yönünü değiştir" disabled={catalogScope === "history"} onClick={() => setSortOrder((current) => current === "desc" ? "asc" : "desc")} type="button"><Icon name="sort" size={16} />{sortOrder === "desc" ? "Azalan" : "Artan"}</button><button className="apply-button" onClick={() => loadCatalog(1)} type="button">Uygula</button></div></div>
              </section>

              {catalogError && <div className="error-banner"><Icon name="warning" size={17} />{catalogError}<button onClick={() => loadCatalog(catalog.page)} type="button">Tekrar dene</button></div>}

              {catalogScope !== "live" && <section className="panel history-results-panel">
                <div className="panel-heading history-results-heading"><div><span>TARİHSEL HİBRİT ARAMA</span><h2>Arşiv kesitleri</h2><p>{formatDate(catalogDateRange.dateFrom)} — {formatDate(catalogDateRange.dateTo)} · Sonuçlar semantik ve tam metin skoruyla sıralanır.</p></div><span className={historyLoading ? "loading-pill active" : "loading-pill"}><i />{historyLoading ? "Arşiv taranıyor" : `${historicalResults.length} eşleşme`}</span></div>
                {query.trim().length < 2 ? <div className="history-search-prompt"><span><Icon name="clock" size={25} /></span><div><strong>{formatNumber(historyOverview.searchable_document_count)} aranabilir tarihsel kayıt hazır.</strong><p>Arşivi taramak için en az iki karakterli bir ürün, kampanya, banka veya koşul yazın.</p></div></div> : historicalResults.length ? <div className="history-result-grid">{historicalResults.map((item) => <article key={item.archive_key}><div className="history-result-top"><span className="history-rank">{String(item.rank).padStart(2, "0")}</span><div><strong>{item.page_title ?? "Başlıksız arşiv belgesi"}</strong><small>{shortBank(item.bank_name)}</small></div><time>{formatDate(item.snapshot_date)}</time></div><p>{item.content}</p><div className="history-result-bottom"><span className="type-pill">{friendlyCode(item.product_type_code)}</span><span>Hibrit skor <b>{item.hybrid_score.toFixed(4)}</b></span><a href={item.archive_url ?? item.source_url ?? "#"} rel="noreferrer" target="_blank">Arşiv kaynağı <Icon name="external" size={13} /></a></div></article>)}</div> : !historyLoading && <div className="empty-state"><span><Icon name="clock" size={26} /></span><h3>Bu dönemde tarihsel eşleşme bulunamadı.</h3><p>Daha geniş bir dönem veya daha genel bir sorgu seçin.</p></div>}
              </section>}

              {catalogScope !== "history" && <section className="panel catalog-panel">
                <div className="catalog-toolbar"><div><strong>{formatNumber(catalog.total)} sonuç</strong><span>Sayfa {catalog.page}/{Math.max(catalog.page_count, 1)}</span></div><span className={catalogLoading ? "loading-pill active" : "loading-pill"}><i />{catalogLoading ? "Veritabanı taranıyor" : "Sonuçlar hazır"}</span></div>
                <div className={catalogLoading ? "catalog-table loading" : "catalog-table"}>
                  <div className="catalog-head"><span>Banka ve ürün</span><span>Tür</span><span>Bilgi kapsamı</span><span>Güven</span><span>Güncelleme</span><span /></div>
                  {catalog.items.map((item) => (
                    <button className="catalog-row" key={item.document_id} onClick={() => openDetail(item.document_id)} type="button">
                      <span className="catalog-title"><BankLogo name={item.bank_name} /><span><strong>{item.page_title ?? "Başlıksız belge"}</strong><small>{shortBank(item.bank_name)} · Belge #{item.document_id}</small></span></span>
                      <span><em className="type-pill">{friendlyCode(item.campaign_type_code, item.campaign_type)}</em></span>
                      <span className="fact-coverage"><b>{item.fact_count}</b><span>{item.fact_types.slice(0, 2).map((fact) => factLabels[fact] ?? fact).join(" · ") || "Yapılandırılmış alan yok"}</span></span>
                      <span className="score-cell"><b>{scoreLabel(item.confidence)}</b><i><u style={{ width: `${Math.round((item.confidence ?? 0) * 100)}%` }} /></i></span>
                      <span className="date-cell">{formatDate(item.updated_at)}</span>
                      <span><Icon name="chevron" size={16} /></span>
                    </button>
                  ))}
                  {!catalog.items.length && <div className="empty-state"><span><Icon name="search" size={26} /></span><h3>Bu filtrelerle kayıt bulunamadı.</h3><p>Ürün türünü veya güven eşiğini değiştirerek tekrar deneyin.</p><button onClick={resetFilters} type="button">Filtreleri temizle</button></div>}
                </div>
                <div className="pagination"><button disabled={catalog.page <= 1 || catalogLoading} onClick={() => loadCatalog(catalog.page - 1)} type="button">Önceki</button><span>{Array.from({ length: Math.min(catalog.page_count, 5) }, (_, index) => { const page = Math.min(Math.max(1, catalog.page - 2), Math.max(1, catalog.page_count - 4)) + index; return page <= catalog.page_count ? <button className={page === catalog.page ? "active" : ""} key={page} onClick={() => loadCatalog(page)} type="button">{page}</button> : null; })}</span><button disabled={catalog.page >= catalog.page_count || catalogLoading} onClick={() => loadCatalog(catalog.page + 1)} type="button">Sonraki</button></div>
              </section>}
            </div>
          )}

          {view === "compare" && (
            <div className="view-stack">
              <section className="comparison-mode-bar panel">
                <div className="scope-copy"><span><Icon name={compareScope === "history" ? "clock" : "database"} size={18} /></span><div><strong>{compareScope === "history" ? "Tarihsel değişim analizi" : "Güncel ürün karşılaştırması"}</strong><small>{compareScope === "history" ? "Dönemin başlangıç ve bitiş kesitlerini karşılaştırır." : "Canlı ürün belgelerindeki en güncel koşulları gösterir."}</small></div></div>
                <div className="segmented-control compact"><button className={compareScope === "live" ? "active" : ""} onClick={() => { setCompareScope("live"); setComparison(null); setComparisonBaseline(null); }} type="button">Güncel</button><button className={compareScope === "history" ? "active" : ""} onClick={() => { setCompareScope("history"); setComparison(null); setComparisonBaseline(null); }} type="button">Tarihsel</button></div>
                {/* Katalogdaki gibi: cipleri sokup cikarmak ustteki kontrolu yerinden oynatiyordu. */}
                <div className={compareScope === "history" ? "period-chips featured" : "period-chips featured hidden"} aria-label="Karşılaştırma dönemi">{historyPeriodOptions.map((period) => <button className={comparePeriod === period.value ? "active" : ""} key={period.value} onClick={() => { setComparePeriod(period.value); setComparison(null); setComparisonBaseline(null); }} type="button">{period.label}</button>)}</div>
              </section>
              <section className="compare-config panel">
                <div className="config-step"><span>1</span><div><strong>Ürün türünü seçin</strong><small>Veritabanındaki sınıflandırılmış ürünler</small></div><select onChange={(event) => { setSelectedProduct(event.target.value); setComparison(null); }} value={selectedProduct}>{options.campaign_types.map((item) => <option key={item.code} value={item.code}>{friendlyCode(item.code, item.label)} · {item.document_count} belge</option>)}</select></div>
                <div className="config-divider" />
                <div className="config-step banks-step"><span>2</span><div><strong>Bankaları belirleyin</strong><small>Varsayılan: verisi bulunan tüm kurumlar</small></div><div className="bank-selector"><button className={!selectedBanks.length ? "selected all-banks" : "all-banks"} onClick={() => setSelectedBanks([])} type="button"><Icon name="building" size={15} />Tüm bankalar ({options.banks.length}){!selectedBanks.length && <Icon name="check" size={13} />}</button>{options.banks.map((bank) => <button className={selectedBanks.includes(bank) ? "selected" : ""} key={bank} onClick={() => toggleBank(bank)} type="button"><BankLogo name={bank} />{shortBank(bank)}{selectedBanks.includes(bank) && <Icon name="check" size={13} />}</button>)}</div></div>
                <div className="config-actions"><span>{selectedBanks.length ? `${selectedBanks.length} banka seçildi` : `Tüm ${options.banks.length} banka taranacak`} · {compareScope === "history" ? activePeriodLabel : "güncel görünüm"}</span><button disabled={!selectedProduct || compareLoading} onClick={runComparison} type="button">{compareLoading ? "Veriler hazırlanıyor…" : compareScope === "history" ? "Dönem değişimini hesapla" : "Tüm bankaları karşılaştır"}<Icon name="arrow" size={17} /></button></div>
              </section>

              {comparison && <section className="panel matrix-panel">
                <div className="panel-heading matrix-heading"><div><span>{compareScope === "history" ? "TARİHSEL BANKA PANOSU" : "BANKA KARŞILAŞTIRMA PANOSU"}</span><h2>{friendlyCode(comparison.campaign_type_code, comparison.campaign_type)}</h2><p>{compareScope === "history" ? `${formatDate(periodDates(comparePeriod, historyOverview).dateFrom)} — ${formatDate(periodDates(comparePeriod, historyOverview).dateTo)} aralığındaki güvenli kesitler.` : `${comparison.count} kaynak belge banka bazında birleştirildi; verisi bulunan bütün kurumlar gösteriliyor.`}</p></div><div className="result-heading-actions"><label>Sırala<select onChange={(event) => setCompareSort(event.target.value)} value={compareSort}><option value="confidence">En yüksek güven</option><option value="facts">En fazla bilgi</option><option value="bank">Banka adı</option></select></label><div className="segmented-control compact"><button className={comparisonView === "cards" ? "active" : ""} onClick={() => setComparisonView("cards")} type="button">Kartlar</button><button className={comparisonView === "matrix" ? "active" : ""} onClick={() => setComparisonView("matrix")} type="button">Matris</button></div></div></div>
                {comparisonTrend && <div className="trend-strip"><div><span><Icon name="chart" size={18} /></span><div><strong>{activePeriodLabel} değişim özeti</strong><small>Dönem başındaki son kesit ile arşiv sonundaki son kesit karşılaştırıldı.</small></div></div><b>{comparisonTrend.changedBanks}<small>kurumda değişim</small></b><b className={comparisonTrend.factDelta >= 0 ? "positive" : "negative"}>{comparisonTrend.factDelta >= 0 ? "+" : ""}{comparisonTrend.factDelta}<small>bilgi alanı farkı</small></b><b>{comparisonTrend.baselineBanks}<small>başlangıç kurumu</small></b></div>}
                {sortedComparison.length ? <>
                  <div className="matrix-summary"><span><b>{sortedComparison.length}</b> banka</span><span><b>{comparison.count}</b> kaynak belge</span><span><b>{activeMatrixFacts.length}</b> karşılaştırılabilir alan</span><span><b>{sortedComparison.reduce((total, item) => total + comparisonFactCount(item), 0)}</b> kaynaklı bulgu</span></div>
                  {comparisonView === "cards" ? <div className="finance-card-list">{sortedComparison.map((item) => <BankComparisonCard historical={compareScope === "history"} item={item} key={item.bank_name} onOpen={() => openComparisonItem(item)} productCode={comparison.campaign_type_code} />)}</div> : <div className="comparison-scroll"><div className="comparison-matrix" style={{ "--compare-count": sortedComparison.length } as React.CSSProperties}>
                    <div className="matrix-corner"><span>Banka bazlı görünüm</span><small>Yatay kaydırarak verisi bulunan bütün bankaları inceleyebilirsiniz.</small></div>
                    {sortedComparison.map((item) => <button className="matrix-bank" key={item.bank_name} onClick={() => openComparisonItem(item)} type="button"><BankLogo name={item.bank_name} /><strong>{shortBank(item.bank_name)}</strong><small>{compareScope === "history" ? item.summary_text : `${item.document_count} belge · ${comparisonFactCount(item)} bulgu`}</small><b>{compareScope === "history" ? `${comparisonFactCount(item)} tarihsel bulgu` : `En yüksek ${scoreLabel(item.confidence)} güven`}</b></button>)}
                    <div className="matrix-row"><div className="matrix-label"><strong>Kaynak ürünler</strong><small>Belge başlıkları</small></div>{sortedComparison.map((item) => <div className="matrix-value source-products" key={`${item.bank_name}-sources`}>{item.page_titles.slice(0, 3).map((title) => <span key={title}>{title}</span>)}{item.page_titles.length > 3 && <em>+{item.page_titles.length - 3} farklı belge</em>}</div>)}</div>
                    {activeMatrixFacts.map((fact) => <div className="matrix-row" key={fact}><div className="matrix-label"><strong>{factLabels[fact] ?? friendlyCode(fact)}</strong><small>{fact}</small></div>{sortedComparison.map((item) => { const values = item.attributes[fact] ?? []; return <div className={values.length ? "matrix-value" : "matrix-value empty"} key={`${item.bank_name}-${fact}`}>{values.length ? <>{values.slice(0, 3).map((value, index) => <span key={`${value.text}-${index}`} title={value.evidence_text ?? undefined}>{value.text}<small>{value.confidence != null ? `%${Math.round(value.confidence * 100)} güven` : value.source}</small></span>)}{values.length > 3 && <em>+{values.length - 3} ek değer</em>}</> : <span>Seçili kaynaklarda yapılandırılmış alan yok</span>}</div>; })}</div>)}
                  </div></div>}
                  {!activeMatrixFacts.length && <div className="matrix-no-facts"><Icon name="warning" size={18} /><div><strong>Bu ürün grubunda yapılandırılmış alan bulunamadı.</strong><p>Kaynak belge başlıkları yine gösteriliyor. Bu durum bankanın ürünü sunmadığı anlamına gelmez.</p></div></div>}
                </> : <div className="empty-state"><span><Icon name="compare" size={28} /></span><h3>Seçiminize uygun kayıt bulunamadı.</h3><p>Başka bir ürün türü veya banka seçerek tekrar deneyin.</p></div>}
              </section>}

              {!comparison && <section className="compare-intro"><article><span><Icon name="database" /></span><strong>Gerçek veritabanı</strong><p>Karşılaştırma kartları sabit metin değil, PostgreSQL kayıtlarından oluşturulur.</p></article><article><span><Icon name="shield" /></span><strong>Kanıt düzeyi</strong><p>Her değer güven puanı ve çıkarıldığı kanıt cümlesiyle saklanır.</p></article><article><span><Icon name="compare" /></span><strong>Eksik veri şeffaflığı</strong><p>Kaynaklarda olmayan koşullar uydurulmaz; açık biçimde işaretlenir.</p></article></section>}
            </div>
          )}

          {view === "assistant" && (
            <div className="view-stack">
              <section className="assistant-mode-bar panel">
                <div className="scope-copy"><span><Icon name={assistantScope === "history" ? "clock" : "spark"} size={18} /></span><div><strong>{assistantScope === "history" ? "Tarihsel RAG" : "Güncel RAG"}</strong><small>{assistantScope === "history" ? `${formatDate(periodDates(assistantPeriod, historyOverview).dateFrom)} — ${formatDate(periodDates(assistantPeriod, historyOverview).dateTo)} arşiv kesitlerinde yanıt arar.` : `${formatNumber(liveDocumentCount)} güncel banka belgesinde yanıt arar.`}</small></div></div>
                <div className="segmented-control compact"><button className={assistantScope === "live" ? "active" : ""} onClick={() => setAssistantScope("live")} type="button">Güncel</button><button className={assistantScope === "history" ? "active" : ""} onClick={() => setAssistantScope("history")} type="button">Tarihsel</button></div>
                <label className="assistant-product-select"><span>Ürün panosu</span><select onChange={(event) => setAssistantProduct(event.target.value)} value={assistantProduct}><option value="auto">Sorudan otomatik algıla</option>{options.campaign_types.map((item) => <option key={item.code} value={item.code}>{friendlyCode(item.code, item.label)}</option>)}</select></label>
                <div className={assistantScope === "history" ? "period-chips featured" : "period-chips featured hidden"}>{historyPeriodOptions.map((period) => <button className={assistantPeriod === period.value ? "active" : ""} key={period.value} onClick={() => setAssistantPeriod(period.value)} type="button">{period.label}</button>)}</div>
              </section>
              <section className="assistant-workspace">
              <div className="assistant-sidebar panel"><div className="assistant-info"><span><Icon name={assistantScope === "history" ? "clock" : "spark"} size={23} /></span><strong>{assistantScope === "history" ? "Tarihsel finans asistanı" : "Kaynaklı RAG asistanı"}</strong><p>BGE-M3 semantik arama ile PostgreSQL tam metin aramasını birleştirir; Qwen yalnızca bulunan kaynaklardan yanıt üretir.</p></div><div className="assistant-tech"><div><span>Embedding</span><strong>BGE-M3</strong></div><div><span>Üretken model</span><strong>{health?.ollama_model ?? "Qwen 3.5"}</strong></div><div><span>Kaynak evreni</span><strong>{assistantScope === "history" ? assistantPeriodLabel : "Güncel"}</strong></div></div><div className="suggestion-list"><span>ÖRNEK SORULAR</span>{assistantQuestions.map((question) => <button key={question} onClick={() => submitChat(question)} type="button">{question}<Icon name="chevron" size={14} /></button>)}</div><button className="new-chat" onClick={() => { setConversation([]); setChatError(null); }} type="button"><Icon name="refresh" size={15} /> Yeni konuşma</button></div>

              <div className="chat-panel panel">
                <div className="chat-top"><div><span className="assistant-avatar">H</span><div><strong>HititFinLex Asistan</strong><small><i /> Kaynak denetimi etkin</small></div></div><span>Yanıtlar finansal tavsiye değildir.</span></div>
                <div className="chat-scroll">
                  {!conversation.length && !chatLoading && <div className="chat-empty"><span><Icon name={assistantScope === "history" ? "clock" : "spark"} size={31} /></span><h2>{assistantScope === "history" ? "Geçmişe kaynaklarıyla sorun." : "Verinizle konuşmaya başlayın."}</h2><p>{assistantScope === "history" ? `${formatNumber(historyOverview.searchable_document_count)} güvenli tarihsel kaydı seçili dönem içinde tarayın.` : "Ürün, banka, vade, kampanya veya masraf sorun. Yanıtla birlikte kullanılan resmî kaynakları da görün."}</p><div>{assistantQuestions.slice(0, 2).map((question) => <button key={question} onClick={() => submitChat(question)} type="button">{question}<Icon name="arrow" size={15} /></button>)}</div></div>}
                  {conversation.map((message) => <article className="conversation-item" key={message.id}><div className="question-bubble"><small>SİZ · {message.scope === "history" ? message.periodLabel : "GÜNCEL"}</small><p>{message.query}</p></div><div className="answer-card"><div className="answer-meta"><span className="assistant-avatar">H</span><div><strong>HititFinLex Asistan</strong><small>{message.model} · {message.sources.length} {message.scope === "history" ? "tarihsel kaynak" : "güncel kaynak"}</small></div></div><div className="answer-copy">{renderAnswer(message.answer, message.sources)}</div>{message.panorama ? <AssistantPanorama data={message.panorama} historical={message.scope === "history"} onOpen={(item) => { if (message.scope === "history") { if (item.source_url) window.open(item.source_url, "_blank", "noopener,noreferrer"); } else openDetail(item.document_id); }} query={message.query} /> : !message.panoramaProduct && <div className="panorama-hint"><Icon name="layers" size={16} /><span>Tüm banka panosu için soruda ürün türünü belirtin veya yukarıdaki “Ürün panosu” alanından seçim yapın.</span></div>}<div className="answer-sources"><span>LLM YANITINDA KULLANILAN KAYNAKLAR</span>{message.sources.map((source) => <article id={`source-${source.source_id}`} key={`${message.id}-${source.source_id}`}><b>{source.source_id}</b><div><strong>{source.page_title ?? "Banka belgesi"}</strong><small>{shortBank(source.bank_name)}{source.snapshot_date ? ` · ${formatDate(source.snapshot_date)}` : ""} · Hibrit skor {source.hybrid_score.toFixed(4)}</small><details><summary>Kanıt metnini göster</summary><p>{source.content}</p></details></div>{(source.archive_url ?? source.source_url) && <a aria-label="Kaynağı aç" href={source.archive_url ?? source.source_url ?? "#"} rel="noreferrer" target="_blank"><Icon name="external" size={15} /></a>}</article>)}</div></div></article>)}
                  {chatLoading && <div className="chat-loading"><span><i /><i /><i /></span><div><strong>Kaynaklar bulunuyor ve yanıt hazırlanıyor…</strong><small>Yerel model ilk yanıtta biraz daha uzun sürebilir.</small></div><button onClick={stopChat} type="button">Durdur</button></div>}
                  {chatError && <div className="chat-error"><Icon name="warning" size={17} /><span>{chatError}</span><button onClick={() => setChatError(null)} type="button"><Icon name="close" size={14} /></button></div>}
                </div>
                <form className="chat-composer" onSubmit={(event) => { event.preventDefault(); submitChat(chatInput); }}><textarea aria-label="Asistana sorun" maxLength={500} onChange={(event) => setChatInput(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); submitChat(chatInput); } }} placeholder="Örn. 2 milyon TL konut finansmanı için hangi koşullar öne çıkıyor?" rows={2} value={chatInput} /><div><span><Icon name="shield" size={13} /> Yanıt yalnızca resmî kaynaklara dayanır</span><span>{chatInput.length}/500</span><button disabled={!chatInput.trim() || chatLoading} type="submit"><Icon name="send" size={17} /> Gönder</button></div></form>
              </div>
              </section>
            </div>
          )}

          {view === "quality" && (
            <div className="view-stack">
              <section className="quality-grid">
                <article className="quality-score panel"><span>GENEL VERİ KAPSAMI</span><div className="score-ring" style={{ "--score": `${dashboard.coverage_percentage * 3.6}deg` } as React.CSSProperties}><div><strong>%{dashboard.coverage_percentage}</strong><small>belgede bilgi var</small></div></div><h2>{formatNumber(dashboard.documents_with_facts)} / {formatNumber(dashboard.document_count)} belge</h2><p>En az bir yapılandırılmış karşılaştırma alanı içeriyor.</p></article>
                <article className="panel quality-summary"><div className="panel-heading"><div><span>İŞLEM DURUMU</span><h2>İnsan denetimi kuyruğu</h2></div><span className={dashboard.pending_document_reviews + dashboard.pending_fact_reviews ? "review-badge warn" : "review-badge"}>{dashboard.pending_document_reviews + dashboard.pending_fact_reviews ? "İnceleme gerekli" : "Kuyruk temiz"}</span></div><div className="quality-kpis"><div><span>Belge incelemesi</span><strong>{dashboard.pending_document_reviews}</strong><small>Sınıflandırma kararı bekliyor</small></div><div><span>Bilgi incelemesi</span><strong>{dashboard.pending_fact_reviews}</strong><small>NER adayı bekliyor</small></div><div><span>Ortalama güven</span><strong>%{Math.round(dashboard.average_confidence * 100)}</strong><small>Tüm sınıflandırmalar</small></div><div><span>Yapılandırılmış bilgi</span><strong>{formatNumber(dashboard.fact_count)}</strong><small>Karşılaştırılabilir alan</small></div></div></article>
              </section>

              <section className="archive-quality panel">
                <div className="archive-quality-title"><span><Icon name="clock" size={20} /></span><div><small>TARİHSEL VERİ KALİTESİ</small><h2>Arşiv, güvenli kayıtlarla aramaya açıldı</h2><p>İnceleme statüsündeki belgeler saklanır ancak arama ve RAG sonuçlarına dahil edilmez.</p></div></div>
                <div><article><span>Toplam arşiv</span><strong>{formatNumber(historyOverview.historical_document_count)}</strong><small>ham tarihsel kesit</small></article><article className="good"><span>Aranabilir</span><strong>{formatNumber(historyOverview.searchable_document_count)}</strong><small>%{historicalCoverage} güvenli kapsam</small></article><article className="warn"><span>İncelemede</span><strong>{formatNumber(historyOverview.review_document_count)}</strong><small>arama dışında tutuluyor</small></article><article><span>Kabul edilen bilgi</span><strong>{formatNumber(historyOverview.historical_fact_count)}</strong><small>kaynaklı NER alanı</small></article><article className="good"><span>Embedding durumu</span><strong>{historyOverview.embedded_chunk_count === historyOverview.historical_chunk_count ? "%100" : `%${Math.round(historyOverview.embedded_chunk_count / Math.max(historyOverview.historical_chunk_count, 1) * 100)}`}</strong><small>{formatNumber(historyOverview.embedded_chunk_count)} / {formatNumber(historyOverview.historical_chunk_count)} parça</small></article></div>
              </section>

              <section className="dashboard-grid quality-lower">
                <article className="panel"><div className="panel-heading"><div><span>ALAN KAPSAMI</span><h2>Bilgi türlerinin dağılımı</h2></div></div><div className="quality-facts">{dashboard.fact_types.map((fact, index) => <div key={fact.code}><span><i className={`quality-dot dot-${index % 6}`} />{fact.label}</span><strong>{formatNumber(fact.count)}</strong><i><u style={{ width: `${Math.min(fact.percentage * 5, 100)}%` }} /></i><small>%{fact.percentage}</small></div>)}</div></article>
                <article className="panel pipeline-panel"><div className="panel-heading"><div><span>HİBRİT MİMARİ</span><h2>Veriden yanıta izlenebilir akış</h2></div></div><div className="pipeline-flow"><div><span><Icon name="document" /></span><div><strong>Birleşik belge havuzu</strong><small>{formatNumber(totalInventoryCount)} güncel + tarihsel kesit</small></div><b>01</b></div><i /><div><span><Icon name="layers" /></span><div><strong>Sınıflandırma + NER</strong><small>{formatNumber(totalFactCount)} kaynaklı bilgi alanı</small></div><b>02</b></div><i /><div><span><Icon name="database" /></span><div><strong>PostgreSQL + pgvector</strong><small>{formatNumber(totalChunkCount)} hibrit arama parçası</small></div><b>03</b></div><i /><div><span><Icon name="spark" /></span><div><strong>Dönem duyarlı Qwen yanıtı</strong><small>Güncel veya tarihsel kaynak seçimi</small></div><b>04</b></div></div></article>
              </section>
            </div>
          )}
        </div>
      </section>

      {(detail || detailLoading) && <div className="drawer-layer"><button aria-label="Belge ayrıntısını kapat" className="drawer-backdrop" onClick={() => setDetail(null)} type="button" /><aside className="detail-drawer">{detailLoading && !detail ? <div className="drawer-loading"><span /><strong>Belge hazırlanıyor…</strong></div> : detail && <><header><div><span>BELGE #{detail.document_id}</span><h2>{detail.page_title ?? "Başlıksız belge"}</h2><p>{shortBank(detail.bank_name)}</p></div><button onClick={() => setDetail(null)} type="button"><Icon name="close" /></button></header><div className="drawer-scroll"><div className="drawer-meta"><span><small>Ürün türü</small><strong>{friendlyCode(detail.campaign_type_code, detail.campaign_type)}</strong></span><span><small>Sınıflandırma güveni</small><strong>{scoreLabel(detail.confidence)}</strong></span><span><small>Çıkarılmış bilgi</small><strong>{detail.facts.length}</strong></span></div>{detail.source_url && <a className="source-link" href={detail.source_url} rel="noreferrer" target="_blank"><Icon name="external" size={16} /> Resmî banka sayfasını aç</a>}<section><div className="drawer-section-title"><span>YAPILANDIRILMIŞ BİLGİLER</span><b>{detail.facts.length}</b></div>{detail.facts.length ? <div className="drawer-facts">{detail.facts.map((fact, index) => <article key={`${fact.fact_type}-${fact.text}-${index}`}><div><span>{fact.label}</span><em>{fact.source}</em></div><strong>{fact.text}</strong>{fact.evidence_text && <p>“{fact.evidence_text}”</p>}<small>{fact.confidence == null ? "Güven bilgisi yok" : `%${Math.round(fact.confidence * 100)} güven`}</small></article>)}</div> : <div className="drawer-empty">Bu belgede yapılandırılmış bilgi bulunamadı.</div>}</section><section><div className="drawer-section-title"><span>BELGE ÖZETİ</span></div><p className="drawer-copy">{detail.summary_text || "Özet bulunmuyor."}</p></section><details className="raw-document"><summary>Ham belge metnini göster</summary><p>{detail.raw_text}</p></details></div></>}</aside></div>}
    </main>
  );
}
