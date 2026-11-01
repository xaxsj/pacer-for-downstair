echo "=== Humanoid.pth 两份 ==="
ls -l /root/autodl-tmp/pacer/Humanoid.pth /root/autodl-tmp/pacer/repo/output/release/pacer/Humanoid.pth 2>/dev/null
md5sum /root/autodl-tmp/pacer/Humanoid.pth /root/autodl-tmp/pacer/repo/output/release/pacer/Humanoid.pth 2>/dev/null
echo ""
echo "=== output/release 内容 ==="
find /root/autodl-tmp/pacer/repo/output/release -type f | head -10
echo ""
echo "=== data/smpl 内容 ==="
ls -l /root/autodl-tmp/pacer/repo/data/smpl 2>/dev/null
echo ""
echo "=== renderings 全部 (仅名字+大小) ==="
ls -1 /root/autodl-tmp/pacer/repo/output/renderings/ | wc -l
ls -lt /root/autodl-tmp/pacer/repo/output/renderings/ | head -4
echo ""
echo "=== 非我们创建的目录 ==="
ls -ld /root/autodl-tmp/Weave /root/autodl-tmp/speedtest.bin 2>/dev/null
echo ""
echo "=== 散落产物统计 ==="
ls -1 /root/autodl-tmp/*.pkl /root/autodl-tmp/*.mp4 2>/dev/null | wc -l
du -ch /root/autodl-tmp/*.pkl /root/autodl-tmp/*.mp4 2>/dev/null | tail -1
echo ""
echo "=== 是否还有进程在跑 ==="
pgrep -af "run.py|python pacer" | grep -v pgrep | head
