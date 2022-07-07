/** safe-ish host dimensions when not running in TTY */
const SAFE_WIDTH  = 80
const SAFE_HEIGHT = 24

/** Terminal host. Provides window metrics for the users terminal. */
export class Host {
    private static _ = process.stdout.on('resize', () => Host.resize())
    public static width  = Host.get_width()
    public static height = Host.get_height()

    /** Resets the terminal width and height. */
    private static resize() {
        this.width  = Host.get_width()
        this.height = Host.get_height()
    }

    /** Returns the current width of the terminal or SAFE_WIDTH if not TTY. */
    private static get_width(): number {
        return process.stdout.isTTY 
            ? process.stdout.columns 
            : SAFE_WIDTH
    }

    /** Returns the current height of the terminal - 1 or SAFE_HEIGHT if not TTY. */
    private static get_height(): number {
        if(process.stdout.isTTY) {
            return (process.stdout.rows >= 3)
                ? process.stdout.rows - 1
                : 1
        } else {
            return SAFE_HEIGHT
        }
    }
}
