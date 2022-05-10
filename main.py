import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GstRtspServer, GObject, Gtk
import spb

class GTK_Main:
    def __init__(self):
        self.make_window()
        self.set_player("rtsp://admin:init123!!@rnd.snuailab.ai:32348/92948fcb-a136-2b25-a2e0-26f6b35eef66")

    def make_window(self):
        window = Gtk.Window(Gtk.WindowType.TOPLEVEL)
        window.set_title("RTSP-Viewer")
        window.set_default_size(500, 400)
        window.connect("destroy", Gtk.main_quit, "WM destroy")

        vbox = Gtk.VBox()
        window.add(vbox)
        self.movie_window = Gtk.DrawingArea()
        vbox.add(self.movie_window)

        hbox0 = Gtk.HBox()
        vbox.pack_start(hbox0, False, False, 0)
        self.entry = Gtk.Entry()
        self.entry.set_text("rtsp://admin:init123!!@rnd.snuailab.ai:32348/92948fcb-a136-2b25-a2e0-26f6b35eef66")
        hbox0.pack_start(self.entry, False, False, 0)

        self.button0 = Gtk.Button("Submit")
        self.button0.connect("clicked", self.set_url)
        hbox0.pack_start(self.button0, False, False, 0)

        hbox = Gtk.HBox()
        vbox.pack_start(hbox, False, False, 0)
        hbox.set_border_width(10)
        hbox.pack_start(Gtk.Label(), False, False, 0)

        self.button = Gtk.Button("Start")
        self.button.connect("clicked", self.start_stop)
        hbox.pack_start(self.button, False, False, 0)
        
        self.button2 = Gtk.Button("Quit")
        self.button2.connect("clicked", self.exit)
        hbox.pack_start(self.button2, False, False, 0)

        hbox.add(Gtk.Label())
        window.show_all()


    def set_player(self, url):
        self.player = Gst.parse_launch (f"uridecodebin uri={url} ! autovideosink")
        bus = self.player.get_bus()
        bus.add_signal_watch()
        bus.enable_sync_message_emission()
        bus.connect("message", self.on_message)
        bus.connect("sync-message::element", self.on_sync_message)

    def set_url(self, widget, data=None):
        url = self.entry.get_text()
        print(url)
        self.set_player(url)
    
    def start_stop(self, w):
        if self.button.get_label() == "Start":
            self.button.set_label("Stop")
            self.player.set_state(Gst.State.PLAYING)
        else:
            self.player.set_state(Gst.State.NULL)
            self.button.set_label("Start")

    def exit(self, widget, data=None):
        Gtk.main_quit()

    def on_message(self, bus, message):
        t = message.type
        if t == Gst.MessageType.EOS:
            self.player.set_state(Gst.State.NULL)
            self.button.set_label("Start")
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err, debug}")
            self.player.set_state(Gst.State.NULL)
            self.button.set_label("Start")

    def on_sync_message(self, bus, message):
        struct = message.get_structure()
        if not struct:
            return
        message_name = struct.get_name()
        if message_name == "prepare-xwindow-id":
            # Assign the viewport
            imagesink = message.src
            imagesink.set_property("force-aspect-ratio", True)
            imagesink.set_xwindow_id(self.movie_window.window.xid)

class MyFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        GstRtspServer.RTSPMediaFactory.__init__(self)
        
    def do_create_element(self, url):
        spec = """
        uridecodebin uri=rtsp://admin:init123'!!'@rnd.snuailab.ai:32348/92948fcb-a136-2b25-a2e0-26f6b35eef66 ! fakesink
        """
        return Gst.parse_launch(spec)

class GstServer():
    def __init__(self):
        #GstRtspServer를 클래스 내에서 선언하고
        self.server = GstRtspServer.RTSPServer()
        #포트를 지정해줍니다.
        #self.server.set_service("3002")
        #팩토리를 생성하는데 이 부분은 위의 MyFactory 클래스에서 설명하겠습니다.
        f = MyFactory()
        #이 팩토리를 공유 할것이라 설정하고
        f.set_shared(True)
        #서버 마운트 포인트를 선언하고
        m = self.server.get_mount_points()
        #마운트 포인트에 주소와 공유할 팩토리를 넣어줍니다.
        m.add_factory("/test", f)
        self.server.attach(None)

if __name__ == '__main__':
    Gst.init(None)
    GTK_Main()
    GObject.threads_init()
    Gtk.main()

    # #루프를 선언하는데 이게 GstRtspServer의 루프입니다.
	# loop = GObject.MainLoop()
	# GObject.threads_init()
	# Gst.init(None)

    # # GstServer 클래스로 서버 설정을 마치고
	# s = GstServer()
    # #서버 루프를 돌립니다.
	# loop.run()