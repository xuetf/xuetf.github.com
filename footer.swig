<div class="copyright" >
  {% set current = date(Date.now(), "YYYY") %}
  &copy; {% if theme.since and theme.since != current %} {{ theme.since }} - {% endif %}
  <span itemprop="copyrightYear">{{ current }}</span>
  <span class="with-love">
    <i class="fa fa-{{ theme.authoricon }}"></i>
  </span>
  <span class="author" itemprop="copyrightHolder">{{ config.author }}</span>
</div>

{% if theme.copyright %}
<div class="powered-by">
  {{ __('footer.powered', '<a class="theme-link" href="https://hexo.io">Hexo</a>') }}
</div>

<div class="theme-info">
  {{ __('footer.theme') }} -
  <a class="theme-link" href="https://github.com/iissnan/hexo-theme-next">
    NexT.{{ theme.scheme }}
  </a>
</div>
{% endif %}


<script type="text/x-mathjax-config">
 MathJax.Hub.Config({"HTML-CSS": { preferredFont: "TeX", availableFonts: ["STIX","TeX"], linebreaks: { automatic:true }, EqnChunk: (MathJax.Hub.Browser.isMobile ? 10 : 50) },
 tex2jax: { inlineMath: [ ["$", "$"], ["\\(","\\)"] ], processEscapes: true, ignoreClass: "tex2jax_ignore|dno",skipTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']},
 TeX: { noUndefined: { attributes: { mathcolor: "red", mathbackground: "#FFEEEE", mathsize: "90%" } }, Macros: { href: "{}" } },
 messageStyle: "none"
 });
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Queue(function() {
 var all = MathJax.Hub.getAllJax(), i;
 for(i=0; i < all.length; i += 1) {
 all[i].SourceElement().parentNode.className += ' has-jax';
 }
 });
</script>
<script type="text/x-mathjax-config">
 MathJax.Hub.Queue(function() {
 var all = MathJax.Hub.getAllJax(), i;
 for(i=0; i < all.length; i += 1) {
 all[i].SourceElement().parentNode.className += ' has-jax';
 }
 });
</script>

<!-- <script charset="utf-8" src="/js/mathjax/2.6-latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script> -->

<script charset="utf-8" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>










