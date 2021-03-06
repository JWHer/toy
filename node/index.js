// import chalk from 'chalk';
// import Image from 'ascii-art-image';

function add() {
    console.log('\x1b[31m%s\x1b[0m', '[clitest Start] 시작합니다.');

    console.log('\x1b[32m%s\x1b[0m',"process.argv[0]: " + process.argv[0]);
    console.log('\x1b[33m%s\x1b[0m',"process.argv[1]: " + process.argv[1]);

    let result = Number(process.argv[2]) + Number(process.argv[3]);

    console.log('\x1b[34m%s\x1b[0m', '[Result] arg1 + arg2 = ', result);
}

function addc() {
    const chalk = require('chalk');
    console.log(chalk.red('[clitest Start] 시작합니다.'));

    console.log(chalk.green("process.argv[0]: " + process.argv[0]));
    console.log(chalk.yellow("process.argv[1]: " + process.argv[1]));

    let result = Number(process.argv[2]) + Number(process.argv[3]);

    console.log(chalk.blue('[Result] arg1 + arg2 = ', result));
}

(() => {
    const Image = require('ascii-art-image');

    var img = new Image({
        filepath: "haha.ha/couple.jpg",
        alphabet:"wide",            // variant1, variant2, variant3, variant4, ultra-wide, wide, hatching, bits, binary, greyscale, blocks
    });
    img.write((err, rendered)=>console.log(rendered));
    // add()
    // addc()
})();
