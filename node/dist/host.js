"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.Host = void 0;
/** safe-ish host dimensions when not running in TTY */
var SAFE_WIDTH = 80;
var SAFE_HEIGHT = 24;
/** Terminal host. Provides window metrics for the users terminal. */
var Host = /** @class */ (function () {
    function Host() {
    }
    /** Resets the terminal width and height. */
    Host.resize = function () {
        this.width = Host.get_width();
        this.height = Host.get_height();
    };
    /** Returns the current width of the terminal or SAFE_WIDTH if not TTY. */
    Host.get_width = function () {
        return process.stdout.isTTY
            ? process.stdout.columns
            : SAFE_WIDTH;
    };
    /** Returns the current height of the terminal - 1 or SAFE_HEIGHT if not TTY. */
    Host.get_height = function () {
        if (process.stdout.isTTY) {
            return (process.stdout.rows >= 3)
                ? process.stdout.rows - 1
                : 1;
        }
        else {
            return SAFE_HEIGHT;
        }
    };
    Host._ = process.stdout.on('resize', function () { return Host.resize(); });
    Host.width = Host.get_width();
    Host.height = Host.get_height();
    return Host;
}());
exports.Host = Host;
//# sourceMappingURL=host.js.map