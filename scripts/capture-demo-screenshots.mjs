import { spawn } from "node:child_process";
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const scriptDir = dirname(fileURLToPath(import.meta.url));
const projectRoot = resolve(scriptDir, "..");
const outputDir = join(projectRoot, "docs", "ekran-goruntuleri");
const appUrl = process.env.HITITFINLEX_DEMO_URL ?? "http://localhost:3000";
const edgePath =
  process.env.EDGE_PATH ??
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe";

const pause = (milliseconds) => new Promise((resolvePause) => setTimeout(resolvePause, milliseconds));

class CdpClient {
  constructor(url) {
    this.nextId = 1;
    this.pending = new Map();
    this.socket = new WebSocket(url);
    this.ready = new Promise((resolveReady, rejectReady) => {
      this.socket.addEventListener("open", resolveReady, { once: true });
      this.socket.addEventListener("error", rejectReady, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const payload = JSON.parse(event.data);
      if (!payload.id) return;
      const pendingRequest = this.pending.get(payload.id);
      if (!pendingRequest) return;
      this.pending.delete(payload.id);
      if (payload.error) pendingRequest.reject(new Error(payload.error.message));
      else pendingRequest.resolve(payload.result);
    });
  }

  async send(method, params = {}) {
    await this.ready;
    const id = this.nextId++;
    const response = new Promise((resolveResponse, rejectResponse) => {
      this.pending.set(id, { resolve: resolveResponse, reject: rejectResponse });
    });
    this.socket.send(JSON.stringify({ id, method, params }));
    return response;
  }

  close() {
    this.socket.close();
  }
}

async function waitForDevToolsPort(profileDir) {
  const activePortFile = join(profileDir, "DevToolsActivePort");
  for (let attempt = 0; attempt < 100; attempt += 1) {
    try {
      const [port] = (await readFile(activePortFile, "utf8")).trim().split(/\r?\n/);
      if (port) return Number(port);
    } catch {
      // Edge creates this file after its headless profile is initialized.
    }
    await pause(100);
  }
  throw new Error("Edge DevTools portu zamanında açılamadı.");
}

async function evaluate(client, expression) {
  const result = await client.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true,
  });
  if (result.exceptionDetails) {
    throw new Error(result.exceptionDetails.text ?? "Tarayıcı değerlendirmesi başarısız.");
  }
  return result.result?.value;
}

async function clickNavigation(client, label, expectedHeading) {
  const clicked = await evaluate(
    client,
    `(() => {
      const target = [...document.querySelectorAll("nav button")]
        .find((button) => button.textContent?.includes(${JSON.stringify(label)}));
      if (!target) return false;
      target.click();
      return true;
    })()`,
  );
  if (!clicked) throw new Error(`Gezinme düğmesi bulunamadı: ${label}`);
  for (let attempt = 0; attempt < 40; attempt += 1) {
    const ready = await evaluate(
      client,
      `document.body?.innerText.includes(${JSON.stringify(expectedHeading)}) ?? false`,
    );
    if (ready) {
      await evaluate(client, "window.scrollTo({ top: 0, behavior: 'instant' })");
      await evaluate(
        client,
        "new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))",
      );
      await client.send("Page.bringToFront");
      return;
    }
    await pause(100);
  }
  throw new Error(`Görünüm zamanında açılmadı: ${label}`);
}

async function waitForSelector(client, selector, timeoutMs = 30000) {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    if (await evaluate(client, `Boolean(document.querySelector(${JSON.stringify(selector)}))`)) return;
    await pause(250);
  }
  throw new Error(`Öğe zamanında görünmedi: ${selector}`);
}

async function clickButtonByText(client, label) {
  const clicked = await evaluate(
    client,
    `(() => {
      const target = [...document.querySelectorAll("button")]
        .find((button) => button.textContent?.trim().includes(${JSON.stringify(label)}));
      if (!target) return false;
      target.click();
      return true;
    })()`,
  );
  if (!clicked) throw new Error(`Düğme bulunamadı: ${label}`);
}

async function focusElement(client, selector, offset = 105) {
  await evaluate(
    client,
    `(() => {
      const target = document.querySelector(${JSON.stringify(selector)});
      if (!target) return false;
      target.scrollIntoView({ block: "start", behavior: "instant" });
      window.scrollBy(0, -${offset});
      return true;
    })()`,
  );
  await evaluate(
    client,
    "new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))",
  );
  await pause(500);
}

// Argüman verilirse yalnızca eşleşen görüntüler yeniden yazılır; böylece tek
// bir ekranı tazelerken diğer dosyalar baytı baytına aynı kalır.
// Örn: node scripts/capture-demo-screenshots.mjs akilli-asistan
const requestedShots = new Set(process.argv.slice(2));

async function capture(client, filename) {
  if (requestedShots.size > 0 && ![...requestedShots].some((name) => filename.includes(name))) {
    process.stdout.write(`Atlandı: ${filename}\n`);
    return;
  }
  const { data } = await client.send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: false,
    fromSurface: true,
  });
  await writeFile(join(outputDir, filename), Buffer.from(data, "base64"));
  process.stdout.write(`Yazıldı: ${filename}\n`);
}

// Asistan görünümü sayfa yüksekliğini tam kaplar ve kendi iç kaydırmasını
// kullanır; sayfayı kaydırmak üst çubuğu kırpıyordu. Bunun yerine sohbet
// listesini kaydırıp yanıtı çerçeveye oturtuyoruz.
async function focusChatAnswer(client) {
  const aligned = await evaluate(
    client,
    `(() => {
      window.scrollTo({ top: 0, behavior: "instant" });
      const scroller = document.querySelector(".chat-scroll");
      const card = document.querySelector(".conversation-item");
      if (!scroller || !card) return false;
      scroller.scrollTop +=
        card.getBoundingClientRect().top - scroller.getBoundingClientRect().top - 10;
      return true;
    })()`,
  );
  if (!aligned) throw new Error("Sohbet yanıtı çerçeveye hizalanamadı.");
  await evaluate(
    client,
    "new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)))",
  );
  await pause(600);
}

async function main() {
  await mkdir(outputDir, { recursive: true });
  const profileDir = await mkdtemp(join(tmpdir(), "hititfinlex-demo-"));
  const edge = spawn(
    edgePath,
    [
      "--headless=new",
      "--disable-gpu",
      "--hide-scrollbars",
      "--no-first-run",
      "--remote-debugging-port=0",
      `--user-data-dir=${profileDir}`,
      "--window-size=1920,1080",
      appUrl,
    ],
    { stdio: "ignore", windowsHide: true },
  );

  let client;
  try {
    const port = await waitForDevToolsPort(profileDir);
    const targets = await fetch(`http://127.0.0.1:${port}/json/list`).then((response) => response.json());
    const page = targets.find((target) => target.type === "page");
    if (!page?.webSocketDebuggerUrl) throw new Error("Edge sayfa hedefi bulunamadı.");

    client = new CdpClient(page.webSocketDebuggerUrl);
    await client.send("Page.enable");
    await client.send("Runtime.enable");
    await client.send("Emulation.setDeviceMetricsOverride", {
      width: 1920,
      height: 1080,
      deviceScaleFactor: 1,
      mobile: false,
    });
    await client.send("Page.navigate", { url: appUrl });
    await pause(3000);

    const bodyText = await evaluate(client, "document.body?.innerText ?? ''");
    if (!bodyText.includes("HititFinLex") || bodyText.includes("Internal Server Error")) {
      throw new Error("HititFinLex arayüzü ekran görüntüsü için hazır değil.");
    }

    // Baglanti rozeti arayuzden kaldirildi; hazir olmayi artik gercek veriden
    // anliyoruz: ilk KPI kartindaki toplam envanter sayisi dolduysa API canli.
    let inventoryLabel = "";
    for (let attempt = 0; attempt < 90; attempt += 1) {
      inventoryLabel = await evaluate(
        client,
        "document.querySelector('.kpi-card strong')?.textContent?.trim() ?? ''",
      );
      if (inventoryLabel && inventoryLabel !== "0") break;
      await pause(500);
    }
    if (!inventoryLabel || inventoryLabel === "0") {
      throw new Error("Frontend canlı API'den veri alamadı; ekran görüntüsü boş olurdu.");
    }
    process.stdout.write(`Toplam veri envanteri: ${inventoryLabel}\n`);

    await capture(client, "01-genel-bakis.png");
    await clickNavigation(client, "Ürün kataloğu", "Ürün ve belge keşfi");
    await capture(client, "02-urun-katalogu.png");
    await clickNavigation(client, "Karşılaştırma", "Koşulları yan yana görün");
    await clickButtonByText(client, "Tüm bankaları karşılaştır");
    // Sonuc alani varsayilan "kart" gorunumunde .finance-card-list, matris
    // gorunumune gecildiginde .comparison-matrix olarak render edilir.
    await waitForSelector(client, ".finance-card-list, .comparison-matrix", 45000);
    await focusElement(client, ".finance-card-list, .comparison-matrix");
    await capture(client, "03-karsilastirma.png");
    await clickNavigation(client, "Akıllı asistan", "HititFinLex Asistan");
    const quickQuestionClicked = await evaluate(
      client,
      `(() => {
        const target = document.querySelector(".chat-empty button");
        if (!target) return false;
        target.click();
        return true;
      })()`,
    );
    if (!quickQuestionClicked) throw new Error("Asistan hızlı sorusu bulunamadı.");
    await waitForSelector(client, ".conversation-item, .chat-error", 120000);
    const chatError = await evaluate(
      client,
      "document.querySelector('.chat-error')?.textContent?.trim() ?? ''",
    );
    if (chatError) throw new Error(`Asistan demo yanıtı üretilemedi: ${chatError}`);
    await focusChatAnswer(client);
    await capture(client, "04-akilli-asistan.png");
    await clickNavigation(client, "Veri kalitesi", "Kalite ve kapsama görünümü");
    await capture(client, "05-veri-kalitesi.png");
  } finally {
    try {
      await client?.send("Browser.close");
    } catch {
      // The browser may already have closed after a capture error.
    }
    client?.close();
    if (edge.exitCode === null) {
      const exited = new Promise((resolveExit) => edge.once("exit", resolveExit));
      await Promise.race([exited, pause(5000)]);
      if (edge.exitCode === null) edge.kill();
      await Promise.race([exited, pause(2000)]);
    }

    const safeTempRoot = resolve(tmpdir()).toLowerCase();
    const safeProfile = resolve(profileDir).toLowerCase();
    if (!safeProfile.startsWith(`${safeTempRoot}\\`) || !safeProfile.includes("hititfinlex-demo-")) {
      throw new Error(`Geçici profil güvenlik denetimini geçemedi: ${profileDir}`);
    }
    for (let attempt = 0; attempt < 5; attempt += 1) {
      try {
        await rm(profileDir, { recursive: true, force: true });
        break;
      } catch (error) {
        if (attempt === 4) throw error;
        await pause(500);
      }
    }
  }
}

try {
  await main();
  process.exit(0);
} catch (error) {
  process.stderr.write(`${error?.stack ?? error}\n`);
  process.exit(1);
}
