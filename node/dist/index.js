// import chalk from 'chalk';
// import Image from 'ascii-art-image';
function add() {
    console.log('\x1b[31m%s\x1b[0m', '[clitest Start] 시작합니다.');
    console.log('\x1b[32m%s\x1b[0m', "process.argv[0]: " + process.argv[0]);
    console.log('\x1b[33m%s\x1b[0m', "process.argv[1]: " + process.argv[1]);
    var result = Number(process.argv[2]) + Number(process.argv[3]);
    console.log('\x1b[34m%s\x1b[0m', '[Result] arg1 + arg2 = ', result);
}
function addc() {
    var chalk = require('chalk');
    console.log(chalk.red('[clitest Start] 시작합니다.'));
    console.log(chalk.green("process.argv[0]: " + process.argv[0]));
    console.log(chalk.yellow("process.argv[1]: " + process.argv[1]));
    var result = Number(process.argv[2]) + Number(process.argv[3]);
    console.log(chalk.blue('[Result] arg1 + arg2 = ', result));
}
(function () {
    var Image = require('ascii-art-image');
    var img = new Image({
        filepath: "haha.ha/couple.jpg",
        alphabet: "wide", // variant1, variant2, variant3, variant4, ultra-wide, wide, hatching, bits, binary, greyscale, blocks
    });
    // img.write((err, rendered)=>console.log(rendered));
    var art = require('ascii-art');
    art.font('word', 'Doom').lines(1, 6, function (err, rendered) {
        // art.image({
        //     filepath: "haha.ha/couple.jpg",
        //     alphabet: 'solid'
        // }).lines(0, 100).overlay(rendered, {
        //     x: 10, y: 10
        // }, (err, final) => console.log(final));
        img.overlay(rendered, {
            x: 10, y: 10
        }, function (err, final) { return console.log(final); });
    });
    // add()
    // addc()
})();
//# sourceMappingURL=index.js.map