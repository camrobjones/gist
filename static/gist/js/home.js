
Vue.config.delimiters = ["[[", "]]"];

var app = new Vue({
    el: '#app',
    delimiters: ['[[', ']]'],
    data: {
        sentence: 'this is a sentence',
        doc: [],
        tree: [],
        query: "",
    },
    filters: {

    },
    computed: {
    },
    methods: {
        // API calls
        analyse: function() {
            let url = '/gist/analyse';
            let csrftoken = Cookies.get('csrftoken');
            let headers = {'X-CSRFToken': csrftoken};
            axios.post(url,{query: this.query}, {headers: headers})
              .then(response => {
                  console.log(response.data);
                  this.doc = response.data.doc;
                  this.tree = response.data.tree;
            });
        },

        totree: function(node) {
            // If you have children: CRC orth /r/c totree(children) /C
            // else C orth /c
            if (Array.isArray(node)) {
                console.log("array: " + node);
                let out = "<div class='tree-level row'> <div class='col'> <div class='row'>";
                for (subnode of node) {
                    out += this.totree(subnode);
                }
                out += "</div> </div> </div>";
                return out;
            }
            else {
                console.log("node: " + node);
                return "<div class='tree-node col'>" + node + "</div>";
                // return node;
            }
        }

    }
});