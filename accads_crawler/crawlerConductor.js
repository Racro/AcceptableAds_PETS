const os = require('os');
const cores = os.cpus().length;
const chalk = require('chalk').default;
const ENABLE_CHALK = false;
chalk.enabled = ENABLE_CHALK;
const async = require('async');
const crawl = require('./crawler');
// const crawl = require('./crawler_auth');
const URL = require('url').URL;
const {createTimer} = require('./helpers/timer');
const createDeferred = require('./helpers/deferred');
const downloadCustomChromium = require('./helpers/downloadCustomChromium');
const {createUniqueUrlName} = require('./helpers/hash');
// eslint-disable-next-line no-unused-vars
const BaseCollector = require('./collectors/BaseCollector');
const notABot = require('./helpers/notABot');

const MAX_NUMBER_OF_CRAWLERS = 1;// by trial and error there seems to be network bandwidth issues with more than 38 browsers. 
const MAX_NUMBER_OF_RETRIES = 2;

/**
 * @param {string} urlString 
 * @param {BaseCollector[]} dataCollectors
 * @param {function} log 
 * @param {boolean} filterOutFirstParty
 * @param {function(URL, import('./crawler').CollectResult): void} dataCallback 
 * @param {boolean} emulateMobile
 * @param {string} proxyHost
 * @param {boolean} antiBotDetection
 * @param {string} executablePath
 * @param {number} maxLoadTimeMs
 * @param {number} extraExecutionTimeMs
 * @param {Object.<string, string>} collectorFlags
 * @param {string} outputPath
 * @param {string} extension
 */
async function crawlAndSaveData(urlString, dataCollectors, log, filterOutFirstParty, dataCallback, emulateMobile, proxyHost, antiBotDetection, executablePath, maxLoadTimeMs, extraExecutionTimeMs, collectorFlags, outputPath, urlHash, extension) {
    const url = new URL(urlString);
    /**
     * @type {function(...any):void} 
     */
    const prefixedLog = (...msg) => log(chalk.gray(`${new Date().toUTCString()} ${createUniqueUrlName(url)}:`), ...msg);

    const data = await crawl(url, {
        log: prefixedLog,
        // @ts-ignore
        collectors: dataCollectors.map(collector => new collector.constructor()),
        filterOutFirstParty,
        emulateMobile,
        proxyHost,
        runInEveryFrame: antiBotDetection ? notABot : undefined,
        executablePath,
        maxLoadTimeMs,
        extraExecutionTimeMs,
        collectorFlags,
        outputPath,
        urlHash,
        extension
    });

    dataCallback(url, data);
}

/**
 * @param {{urls: Array<string|{url:string,dataCollectors?:BaseCollector[]}>, dataCallback: function(URL, import('./crawler').CollectResult): void, dataCollectors?: BaseCollector[], failureCallback?: function(string, Error): void, numberOfCrawlers?: number, logFunction?: function, filterOutFirstParty: boolean, emulateMobile: boolean, proxyHost: string, antiBotDetection?: boolean, chromiumVersion?: string, maxLoadTimeMs?: number, extraExecutionTimeMs?: number, collectorFlags?: Object.<string, boolean>, outputPath:string, extension:string}} options
 */
module.exports = async options => {
    const deferred = createDeferred();
    const log = options.logFunction || (() => {});
    const failureCallback = options.failureCallback || (() => {});

    let numberOfCrawlers = options.numberOfCrawlers || Math.floor(cores * 0.8);
    numberOfCrawlers = Math.min(MAX_NUMBER_OF_CRAWLERS, numberOfCrawlers, options.urls.length);

    // Increase number of listeners so we have at least one listener for each async process
    if (numberOfCrawlers > process.getMaxListeners()) {
        process.setMaxListeners(numberOfCrawlers + 1);
    }
    log(chalk.cyan(`Number of crawlers: ${numberOfCrawlers}\n`));
    process.on('uncaughtException', err => {
        // Ignore the error, exiting the process terminates the crawl
        log(`ERROR: uncaughtException - START: ${err.stack}`);
        log('ERROR: uncaughtException - END'); // mark the end of the multiline error message
    });
    // // the following is untested
    // process.on('unhandledRejection', (reason, p) => {
    //     log('ERROR: unhandledRejection at promise', reason, p);
    // })

    /**
     * @type {string}
     */
    let executablePath;
    if (options.chromiumVersion) {
        executablePath = await downloadCustomChromium(log, options.chromiumVersion);
    }

    async.eachOfLimit(options.urls, numberOfCrawlers, (urlItem, idx, callback) => {
        const urlString = (typeof urlItem === 'string') ? urlItem : urlItem.url;
        let dataCollectors = options.dataCollectors;

        // there can be a different set of collectors for every item
        if ((typeof urlItem !== 'string') && urlItem.dataCollectors) {
            dataCollectors = urlItem.dataCollectors;
        }
        const url = new URL(urlString);
        const urlHash = createUniqueUrlName(url);
        log(chalk.cyan(`Processing entry #${Number(idx) + 1} (${urlString}) Hash: ${urlHash}.`));
        const timer = createTimer();

        const task = crawlAndSaveData.bind(null, urlString, dataCollectors, log, options.filterOutFirstParty, options.dataCallback, options.emulateMobile, options.proxyHost, (options.antiBotDetection !== false), executablePath, options.maxLoadTimeMs, options.extraExecutionTimeMs, options.collectorFlags, options.outputPath, urlHash, options.extension);

        async.retry(MAX_NUMBER_OF_RETRIES, task, err => {
            if (err) {
                log(err)
                log(chalk.red(`Max number of retries (${MAX_NUMBER_OF_RETRIES}) exceeded for "${urlString}" with err: ${err}`));
                failureCallback(urlString, err);
            } else {
                log(chalk.cyan(`Processing "${urlString}" took ${timer.getElapsedTime()}s.`));
            }

            callback();
        });
    }, err => {
        if (err) {
            deferred.reject(err);
        } else {
            deferred.resolve();
        }
    });

    await deferred.promise;
};
