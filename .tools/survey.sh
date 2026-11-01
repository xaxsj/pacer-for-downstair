echo "############ TOP LEVEL ############"
du -sh /root/autodl-tmp/* 2>/dev/null | sort -hr
echo ""
echo "############ FREE SPACE ############"
df -h /root/autodl-tmp | tail -1
echo ""
echo "############ repo depth-1 ############"
du -sh /root/autodl-tmp/pacer/repo/* 2>/dev/null | sort -hr
echo ""
echo "############ repo depth-2 (data/output/experiment) ############"
du -sh /root/autodl-tmp/pacer/repo/data/* 2>/dev/null | sort -hr
echo "---"
du -sh /root/autodl-tmp/pacer/repo/output/* 2>/dev/null | sort -hr
echo "---"
ls -la /root/autodl-tmp/pacer/repo/experiment_result 2>/dev/null
echo ""
echo "############ caches / pycache ############"
find /root/autodl-tmp -name __pycache__ -type d 2>/dev/null | wc -l
du -sh $(find /root/autodl-tmp -name __pycache__ -type d 2>/dev/null | head -1) 2>/dev/null
find /root/autodl-tmp -name "*.pyc" 2>/dev/null | wc -l
echo "torch_extensions:"
du -sh /root/.cache/torch_extensions 2>/dev/null
echo ""
echo "############ recordings ############"
ls -la /root/autodl-tmp/pacer/repo/output/recordings/ 2>/dev/null | head -20
echo ""
echo "############ renderings ############"
ls -la /root/autodl-tmp/pacer/repo/output/renderings/ 2>/dev/null | head -20
echo ""
echo "############ /tmp ############"
ls -la /tmp | head -25
echo "--- /tmp/pacer_tmp ---"
du -sh /tmp/pacer_tmp 2>/dev/null; ls -la /tmp/pacer_tmp 2>/dev/null | head
echo ""
echo "############ tarballs ############"
ls -la /root/autodl-tmp/pacer/*.tar.gz /root/autodl-tmp/pacer/*.zip /root/autodl-tmp/pacer/*.pth 2>/dev/null
echo ""
echo "############ repo root loose files ############"
ls -la /root/autodl-tmp/pacer/repo | head -40
