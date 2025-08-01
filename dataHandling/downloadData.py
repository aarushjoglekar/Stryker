#Imports
import SoccerNet
from SoccerNet.Downloader import SoccerNetDownloader

#Download data from SoccerNet
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="SoccerNet/labels")
mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadGames(files=["1_224p.mkv", "2_224p.mkv"], split=["train", "valid", "test"])
mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])