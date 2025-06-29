/* eslint-disable max-lines */
const puppeteer = require('puppeteer');
const StealthPlugin = require('puppeteer-extra-plugin-stealth');
const puppeteerExtra = require('puppeteer-extra');
const chalk = require('chalk').default;
const {createTimer} = require('./helpers/timer');
const wait = require('./helpers/wait');
const tldts = require('tldts');
const fs = require('fs');
var Xvfb = require('xvfb');
// const fingerprintDetection = require('./helpers/fingerprintDetection');
// TODO: try converting back to module.exports
const fpSrc = fs.readFileSync('./helpers/fingerprintDetection.js', 'utf8');

const ENABLE_CHALK = false;
chalk.enabled = ENABLE_CHALK;

const DEFAULT_USER_AGENT = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36';
const MOBILE_USER_AGENT = 'Mozilla/5.0 (Linux; Android 10; Pixel 2 XL) AppleWebKit/537.36 (KHTML, like Gecko)  Chrome/112.0.0.0 Mobile Safari/537.36';

// Ritik
var xvfb_switch = 0;
var xvfb = new Xvfb({
    silent: true,
    reuse: true,
    xvfb_args: ["-screen", "0", '1280x720x24', "-ac"],
});

if (xvfb_switch === 1){
    xvfb.startSync((err)=>{if (err) console.error(err)});
}

const DEFAULT_VIEWPORT = {
    width: 1920,  // px
    height: 1080  // px
};
const MOBILE_VIEWPORT = {
    width: 412,
    height: 691,
    deviceScaleFactor: 2,
    isMobile: true,
    hasTouch: true
};

// for debugging: will launch in window mode instad of headless, open devtools and don't close windows after process finishes
const VISUAL_DEBUG = true;

/**
 * @param {function(...any):void} log
 * @param {string} proxyHost
 * @param {string} executablePath path to chromium executable to use
 * @param {string} extension path to chromium executable to use
 */
function openBrowser(log, proxyHost, executablePath, extension) {
    /**
     * @type {import('puppeteer').BrowserLaunchArgumentOptions}
     */

    var args = null;
    if (xvfb_switch === 1){
        if (extension === 'control'){
            args = {
                args: [
                    // enable FLoC
                    '--no-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--start-maximized',
                    '--user-data-dir=./temp_session/',
                    // '--user-data-dir=./saved_session/',
                    '--display='+xvfb._display,
                ]
            };
        } else {
            args = {
                args: [
                    // enable FLoC
                    '--no-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--start-maximized',
                    `--disable-extensions-except=./extn_src/${extension}_v2`,
                    `--load-extension=./extn_src/${extension}_v2`,
                    // '--user-data-dir=./saved_session/',
                    '--user-data-dir=./temp_session/',
                    '--display='+xvfb._display,

                ]
            };
        }
    } else {
        if (extension === 'control'){
            args = {
                args: [
                    // enable FLoC
                    '--no-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--start-maximized',
                    // '--user-data-dir=./temp_session/',
                    '--user-data-dir=./saved_session/'
                ]
            };
        } else {
            args = {
                args: [
                    // enable FLoC
                    '--no-sandbox',
                    '--disable-web-security',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--start-maximized',
                    `--disable-extensions-except=./extn_src/${extension}_v2`,
                    `--load-extension=./extn_src/${extension}_v2`,
                    '--user-data-dir=./saved_session/',
                    // '--user-data-dir=./temp_session/'
                ]
            };
        }
    }

    if (VISUAL_DEBUG) {
        args.headless = false;
        args.devtools = false;
    }
    if (proxyHost) {
        let url;
        try {
            url = new URL(proxyHost);
        } catch(e) {
            log('Invalid proxy URL');
        }

        args.args.push(`--proxy-server=$
        {proxyHost}`);
        args.args.push(`--host-resolver-rules="MAP * ~NOTFOUND , EXCLUDE ${url.hostname}"`);
    }

    args.executablePath = './chrome-linux/chrome';

    puppeteerExtra.default.use(StealthPlugin());
    return puppeteerExtra.default.launch(args);
}

/**
 * @param {puppeteer.BrowserContext} context
 * @param {URL} url
 * @param {{collectors: import('./collectors/BaseCollector')[];log: function(...any):void;urlFilter: function(string, string):boolean;emulateMobile: boolean;emulateUserAgent: boolean;runInEveryFrame: function():void;maxLoadTimeMs: number;extraExecutionTimeMs: number;collectorFlags: Object<string, string>;outputPath: string;urlHash: string;extension: string;}} data
 * @returns {Promise<CollectResult>}
 */
async function getSiteData(context, url, {
    collectors,
    log,
    urlFilter,
    emulateUserAgent,
    emulateMobile,
    runInEveryFrame,
    maxLoadTimeMs,
    extraExecutionTimeMs,
    collectorFlags,
    outputPath,
    urlHash,
    extension
}) {
    const testStarted = Date.now();

    /**
     * @type {{cdpClient: import('puppeteer').CDPSession, type: string, url: string}[]}
     */
    const targets = [];

    const collectorOptions = {
        context,
        url,
        log,
        collectorFlags,
        outputPath,
        urlHash,
        emulateMobile
    };
    // console.error('00000000000\n');
    for (let collector of collectors) {
        const timer = createTimer();

        try {
            // eslint-disable-next-line no-await-in-loop
            await collector.init(collectorOptions);
            // log(`${collector.id()} init took ${timer.getElapsedTime()}s`);
        } catch (e) {
            log(chalk.yellow(`${collector.id()} init failed`), chalk.gray(e.message), chalk.gray(e.stack));
        }
    }

    let pageTargetCreated = false;

    // initiate collectors for all contexts (main page, web worker, service worker etc.)
    context.on('targetcreated', async target => {
        if (target.type() === 'page' && pageTargetCreated) {
            let newPage = await target.page();
            let adCollector = collectors.find(collector => collector.id() === 'ads');
            try {
                // eslint-disable-next-line no-await-in-loop
                await adCollector.addListener(newPage, url, outputPath, urlHash, context);
            } catch (e) {
                log(chalk.yellow(`${adCollector.id()} failed to attach to "${target.url()}"`), chalk.gray(e.message), chalk.gray(e.stack));
            }
            return;
        }
        // we have already initiated collectors for the main page, so we ignore the first page target
        if (target.type() === 'page' && !pageTargetCreated) {
            pageTargetCreated = true;
            return;
        }

        const timer = createTimer();
        let cdpClient = null;

        try {
            cdpClient = await target.createCDPSession();
            log(`${target.type()} target created in ${timer.getElapsedTime()}s`);
        } catch (e) {
            log(chalk.yellow(`Failed to connect to "${target.url()}"`), chalk.gray(e.message), chalk.gray(e.stack));
            return;
        }

        const simpleTarget = {url: target.url(), type: target.type(), cdpClient};
        targets.push(simpleTarget);

        try {
            // we have to pause new targets and attach to them as soon as they are created not to miss any data
            await cdpClient.send('Target.setAutoAttach', {autoAttach: true, waitForDebuggerOnStart: true});
        } catch (e) {
            log(chalk.yellow(`Failed to set "${target.url()}" up.`), chalk.gray(e.message), chalk.gray(e.stack));
            return;
        }

        for (let collector of collectors) {
            try {
                if(target.type() === 'page' && pageTargetCreated) {
                    log(`Skipping ${collector.id()} for ${target.type()} target since it's ad page.`);
                    continue;
                }
                // eslint-disable-next-line no-await-in-loop
                await collector.addTarget(simpleTarget);
                // log(`${collector.id()} attached to "${target.url()}" in ${timer.getElapsedTime()}s`);
            } catch (e) {
                log(chalk.yellow(`${collector.id()} failed to attach to "${target.url()}"`), chalk.gray(e.message), chalk.gray(e.stack));
            }
        }

        try {
            // resume target when all collectors are ready
            await cdpClient.send('Runtime.enable');
            await cdpClient.send('Runtime.runIfWaitingForDebugger');
        } catch (e) {
            log(chalk.yellow(`Failed to resume target "${target.url()}"`), chalk.gray(e.message), chalk.gray(e.stack));
            return;
        }

        log(`${target.url()} (${target.type()}) context initiated in ${timer.getElapsedTime()}s`);
    });

    // RITIK CHECK

    // console.log(`REACHED HERE`);
    // await new Promise(r => setTimeout(r, 5000));
    // Create a new page in a pristine context.
    // New page in incognito
    // const page = await context.newPage();

    // For logged in crawl - opens page in about:blank tab
    const [page] = await context.pages()
    
    // optional function that should be run on every page (and subframe) in the browser context
    if (runInEveryFrame) {
        page.evaluateOnNewDocument(runInEveryFrame);
        // page.evaluateOnNewDocument(fingerprintDetection);
        page.evaluateOnNewDocument(fpSrc);
    }
    for (let collector of collectors) {
        if(collector.id() === 'fingerprints') {
            try {
                // eslint-disable-next-line no-await-in-loop
                await collector.addListener(page);
            } catch (e) {
                log(chalk.yellow(`${collector.id()} failed to attach to page`), chalk.gray(e.message), chalk.gray(e.stack));
            }
        }
    }

    // We are creating CDP connection before page target is created, if we create it only after
    // new target is created we will miss some requests, API calls, etc.
    const cdpClient = await page.target().createCDPSession();

    // without this, we will miss the initial request for the web worker or service worker file
    await cdpClient.send('Target.setAutoAttach', {autoAttach: true, waitForDebuggerOnStart: true});

    const initPageTimer = createTimer();
    for (let collector of collectors) {
        try {
            // eslint-disable-next-line no-await-in-loop
            await collector.addTarget({url: url.toString(), type: 'page', cdpClient, page, outputPath, urlHash});
        } catch (e) {
            log(chalk.yellow(`${collector.id()} failed to attach to page`), chalk.gray(e.message), chalk.gray(e.stack));
        }
    }
    log(`page context initiated in ${initPageTimer.getElapsedTime()}s`);

    if (emulateUserAgent) {
        await page.setUserAgent(emulateMobile ? MOBILE_USER_AGENT : DEFAULT_USER_AGENT);
    }

    await page.setViewport(emulateMobile ? MOBILE_VIEWPORT : DEFAULT_VIEWPORT);

    // if any prompts open on page load, they'll make the page hang unless closed
    page.on('dialog', dialog => dialog.dismiss());

    // catch and report crash errors
    page.on('error', e => log(chalk.red(e.message)));

    // Ritik - timeput for extension to load in incognito mode/context
    // await new Promise(r => setTimeout(r, 5000));

    let timeout = false;
    try {
        // log the page load time
        const loadTimer = createTimer();
        await page.goto(url.toString(), {timeout: maxLoadTimeMs, waitUntil: 'networkidle0'});
        log(`⏱️ Page load took ${loadTimer.getElapsedTime()}s`);
    } catch (e) {
        if (e instanceof puppeteer.errors.TimeoutError || (e.name && e.name === 'TimeoutError')) {
            log(chalk.yellow('Navigation timeout exceeded.'));

            for (let target of targets) {
                if (target.type === 'page') {
                    // eslint-disable-next-line no-await-in-loop
                    await target.cdpClient.send('Page.stopLoading');
                }
            }
            timeout = true;
        } else {
            throw e;
        }
    }

    for (let collector of collectors) {
        const postLoadTimer = createTimer();
        try {
            // eslint-disable-next-line no-await-in-loop
            await collector.postLoad();
            const elapsed = postLoadTimer.getElapsedTime();
            if (! elapsed.startsWith('0.')) {
                log(`⏱️ ${collector.id()} postLoad took ${elapsed}s`);
            }
        } catch (e) {
            log(chalk.yellow(`${collector.id()} postLoad failed`), chalk.gray(e.message), chalk.gray(e.stack));
        }
    }

    // give website a bit more time for things to settle
    await page.waitForTimeout(extraExecutionTimeMs);
    const finalUrl = page.url();

    /**
     * @type {Object<string, Object>}
     */
    const data = {};

    // var frames = page.frames();

    // for (let frame = 0; frame < frames.length; frame++){
    for (let collector of collectors) {
        const getDataTimer = createTimer();
        try {
            // eslint-disable-next-line no-await-in-loop
            const collectorData = await collector.getData({
                finalUrl,
                urlFilter: urlFilter && urlFilter.bind(null, finalUrl),
                page,
                // page: frames[frame],
                outputPath,
                urlHash,
                context
            });
            data[collector.id()] = collectorData;
            const elapsed = getDataTimer.getElapsedTime();
            if (! elapsed.startsWith('0.')) {
                log(`⏱️ ${collector.id()} data took ${elapsed}s`);
            }
            // log(`getting ${collector.id()} data took ${}s`);
        } catch (e) {
            log(chalk.yellow(`getting ${collector.id()} data failed`), chalk.gray(e.message), chalk.gray(e.stack));
            data[collector.id()] = null;
        }
    }
    // }

    for (let target of targets) {
        try {
            // eslint-disable-next-line no-await-in-loop
            await target.cdpClient.detach();
        } catch (e) {
            // we don't care that much because in most cases an error here means that target already detached
        }
    }

    // Ritik
    // if (!VISUAL_DEBUG) {
    //     await page.close();
    // }

    await new Promise(r => setTimeout(r, 100000));
    await page.close();

    return {
        initialUrl: url.toString(),
        finalUrl,
        timeout,
        testStarted,
        testFinished: Date.now(),
        data
    };
}

/**
 * @param {string} documentUrl
 * @param {string} requestUrl
 * @returns {boolean}
 */
function isThirdPartyRequest(documentUrl, requestUrl) {
    const mainPageDomain = tldts.getDomain(documentUrl);

    return tldts.getDomain(requestUrl) !== mainPageDomain;
}

/**
 * @param {URL} url
 * @param {{collectors?: import('./collectors/BaseCollector')[], log?: function(...any):void, filterOutFirstParty?: boolean, emulateMobile?: boolean, emulateUserAgent?: boolean, proxyHost?: string, browserContext?: puppeteer.BrowserContext, runInEveryFrame?: function():void, executablePath?: string, maxLoadTimeMs?: number, extraExecutionTimeMs?: number, collectorFlags?: Object.<string, string>, outputPath: string, urlHash: string, extension: string}} options
 * @returns {Promise<CollectResult>}
 */
module.exports = async (url, options) => {
    const log = options.log || (() => {});
    const browser = options.browserContext ? null : await openBrowser(log, options.proxyHost, options.executablePath, options.extension);
    // Create a new incognito browser context.
    // const context = options.browserContext || await browser.createIncognitoBrowserContext();
    


    if (options.extension === 'adblock'){
        try {
        // const test = await browser.newPage();
        // // await test.setViewport({ width: 1280, height: 720 });
        // await test.goto('chrome://extensions', { timeout: 20000 });
        // await test.screenshot({
        //     path: 'extension_verification.jpg'
        // });

            // await sleep(10000);
            await new Promise(r => setTimeout(r, 10000));
            const extensionsPage = await browser.newPage();
            await extensionsPage.goto( 'chrome://extensions/' );
            await extensionsPage.screenshot({
                path: './extension_verification.jpg'
            });
            
            // Only for unlogged crawls
            // await extensionsPage.evaluate(`
            // chrome.developerPrivate.getExtensionsInfo().then((extensions) => {
            //     extensions.map((extension) => chrome.developerPrivate.updateExtensionConfiguration({extensionId: extension.id, incognitoAccess: true}));
            // });
            // `);

        } catch (e) {
            console.error('\n00000000000000\n');
            await browser.close();
            throw e;
        };
    } 
    else {
        await new Promise(r => setTimeout(r, 2000));
        // await sleep(2000);
    }

    // try {
    // const test = await browser.newPage();
    // const version = await test.browser().version();
    // console.error(`\n VERSION:${version} \n`);

    // } catch (e) {
    //     await browser.close();
    //     throw e;
    // } 

    // Ritik - If without logging in
    // const context = options.browserContext || await browser.createIncognitoBrowserContext();

    // Ritik - Logging in
    const context = options.browserContext || await browser.defaultBrowserContext();

    // await new Promise(r => setTimeout(r, 5000));


    let data = null;

    const maxLoadTimeMs = options.maxLoadTimeMs || 30000;
    const extraExecutionTimeMs = options.extraExecutionTimeMs || 2500;
    const maxTotalTimeMs = maxLoadTimeMs * 8;

    try {
        data = await wait(getSiteData(context, url, {
            collectors: options.collectors || [],
            log,
            urlFilter: options.filterOutFirstParty === true ? isThirdPartyRequest.bind(null) : null,
            emulateUserAgent: options.emulateUserAgent !== false, // true by default
            emulateMobile: options.emulateMobile,
            runInEveryFrame: options.runInEveryFrame,
            maxLoadTimeMs,
            extraExecutionTimeMs,
            collectorFlags: options.collectorFlags,
            outputPath: options.outputPath,
            urlHash: options.urlHash,
            extension: options.extension,
        }), maxTotalTimeMs);
    } catch(e) {
        log(chalk.red('******************* ERROR: Crawl failed *******************'), e.message, chalk.gray(e.stack));
        throw e;
    } finally {
        // only close the browser if it was created here and not debugging
        // if (browser && !VISUAL_DEBUG) {
        try{
            if (browser) {
                log('Closing the context and the browser');
                // await context.close();
                // log('Context closed');
                await browser.close();
                log('Browser process closed');
            }
        } catch(e){
            log(chalk.red('While closing browser -- '), e.message);
        }
    }

    await xvfb.stopSync();
    return data;
};

/**
 * @typedef {Object} CollectResult
 * @property {string} initialUrl URL from which the crawler began the crawl (as provided by the caller)
 * @property {string} finalUrl URL after page has loaded (can be different from initialUrl if e.g. there was a redirect)
 * @property {boolean} timeout true if page didn't fully load before the timeout and loading had to be stopped by the crawler
 * @property {number} testStarted time when the crawl started (unix timestamp)
 * @property {number} testFinished time when the crawl finished (unix timestamp)
 * @property {import('./helpers/collectorsList').CollectorData} data object containing output from all collectors
*/
