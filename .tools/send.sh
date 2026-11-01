echo "=== Humanoid.pth 涓や唤 ==="
ls -l /root/autodl-tmp/pacer/Humanoid.pth /root/autodl-tmp/pacer/repo/output/release/pacer/Humanoid.pth 2>/dev/null
md5sum /root/autodl-tmp/pacer/Humanoid.pth /root/autodl-tmp/pacer/repo/output/release/pacer/Humanoid.pth 2>/dev/null
echo ""
echo "=== output/release 鍐呭 ==="
find /root/autodl-tmp/pacer/repo/output/release -type f | head -10
echo ""
echo "=== data/smpl 鍐呭 ==="
ls -l /root/autodl-tmp/pacer/repo/data/smpl 2>/dev/null
echo ""
echo "=== renderings 鍏ㄩ儴 (浠呭悕瀛?澶у皬) ==="
ls -1 /root/autodl-tmp/pacer/repo/output/renderings/ | wc -l
ls -lt /root/autodl-tmp/pacer/repo/output/renderings/ | head -4
echo ""
echo "=== 闈炴垜浠垱寤虹殑鐩綍 ==="
ls -ld /root/autodl-tmp/Weave /root/autodl-tmp/speedtest.bin 2>/dev/null
echo ""
echo "=== 鏁ｈ惤浜х墿缁熻 ==="
ls -1 /root/autodl-tmp/*.pkl /root/autodl-tmp/*.mp4 2>/dev/null | wc -l
du -ch /root/autodl-tmp/*.pkl /root/autodl-tmp/*.mp4 2>/dev/null | tail -1
echo ""
echo "=== 鏄惁杩樻湁杩涚▼鍦ㄨ窇 ==="
pgrep -af "run.py|python pacer" | grep -v pgrep | head
