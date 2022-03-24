
var app = new Vue({
  el: '#app',
  delimiters: ["[[", "]]"],
  data: {
    // Settings
    mode: "loading",

    // Explore
    synset: "",
    lemma: "",
    lemmas: [],
    examples: [],
    definition: "",
    hyponyms: [],
    hypernyms: [],

    // Search
    query: "",
    results: [],
  },
  mounted() {
    this.get_synset('entity.n.01');
  },
  methods: {
    get_synset: function(synset_key) {
        this.mode = "loading";
        var queryString = '/gist/synset_data?synset_key=' + synset_key;
        console.log(queryString);
        axios.get(queryString).then(response => {
            console.log(response);
            this.mode = "explore";
            this.synset = response.data;
            // this.lemma = response.data.lemma;
            // this.lemmas = response.data.lemmas;
            // this.examples = response.data.examples;
            // this.definition = response.data.definition;
            // this.hyponyms = response.data.hyponyms;
            // this.hypernyms = response.data.hypernyms;
            
            this.$nextTick(
                // Force update on next tick. TODO: neater solution
                function(){
                    app.$forceUpdate();
                }
            );
        });
    },

    searchLemma: function() {
        this.mode="loading";
        var queryString = '/gist/search_lemma?lemma=' + this.query;
        axios.get(queryString).then(response => {
            console.log(response);
            this.mode = "search";
            this.results = response.data;
        });
    },

    getX: function(id) {
        let el = $('#'+id);
        if (el.length > 0) {
            // + 10 is a hacky solution to something ignoring margins
            return el.position().left + 10 + el.outerWidth() / 2;
        }
        return 0;
    },
    getY: function(id) {
        let el = $('#'+id);
        if (el.length > 0) {
            return el.position().top + 10 + el.outerHeight() / 2;
        }
        return 0;
    },
    getHeight: function(id) {
        let el = $('#'+id);
        return el.height();
    }

  }
});