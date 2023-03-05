function simulate(element, eventName)
{
    var options = extend(defaultOptions, arguments[2] || {});
    var oEvent, eventType = null;

    for (var name in eventMatchers)
    {
        if (eventMatchers[name].test(eventName)) { eventType = name; break; }
    }

    if (!eventType)
        throw new SyntaxError('Only HTMLEvents and MouseEvents interfaces are supported');

    if (document.createEvent)
    {
        oEvent = document.createEvent(eventType);
        if (eventType == 'HTMLEvents')
        {
            oEvent.initEvent(eventName, options.bubbles, options.cancelable);
        }
        else
        {
            oEvent.initMouseEvent(eventName, options.bubbles, options.cancelable, document.defaultView,
            options.button, options.pointerX, options.pointerY, options.pointerX, options.pointerY,
            options.ctrlKey, options.altKey, options.shiftKey, options.metaKey, options.button, element);
        }
        element.dispatchEvent(oEvent);
    }
    else
    {
        options.clientX = options.pointerX;
        options.clientY = options.pointerY;
        var evt = document.createEventObject();
        oEvent = extend(evt, options);
        element.fireEvent('on' + eventName, oEvent);
    }
    return element;
}

function extend(destination, source) {
    for (var property in source)
      destination[property] = source[property];
    return destination;
}

var eventMatchers = {
    'HTMLEvents': /^(?:load|unload|abort|error|select|change|submit|reset|focus|blur|resize|scroll)$/,
    'MouseEvents': /^(?:click|dblclick|mouse(?:down|up|over|move|out))$/
}
var defaultOptions = {
    pointerX: 0,
    pointerY: 0,
    button: 0,
    ctrlKey: false,
    altKey: false,
    shiftKey: false,
    metaKey: false,
    bubbles: true,
    cancelable: true
}


function buildVegaSpec(data) {
    let spec = {
      "$schema": "https://vega.github.io/schema/vega/v5.json",
      "width": 600,
      "height": 600,
      "padding": 5,
      "data": [
        {
          "name": "source",
          "values": data,
          "transform": [
            {
              "type": "filter",
              "expr": "datum['word'] != null && datum['x'] != null && datum['y'] != null"
            }
          ]
        }
      ],
      "signals": [
          {
          "name": "width",
          "value": "600",
          "on": [
            {
              "events": {
                "source": "window",
                "type": "resize"
              },
              "update": "containerSize()[0]"
            },
            {
              "events": {
                "source": "window",
                "type": "load"
              },
              "update": "containerSize()[0]"
            }
          ]
        },
        {
          "name": "height",
          "value": "600",
          "on": [
            {
              "events": {
                "source": "window",
                "type": "resize"
              },
              "update": "containerSize()[1]"
            },
            {
              "events": {
                "source": "window",
                "type": "load"
              },
              "update": "containerSize()[0]"
            }
          ]
        }
      ],
      "scales": [
        {
          "name": "x",
          "type": "linear",
          "round": true,
          "nice": true,
          "zero": true,
          "domain": {"data": "source", "field": "x"},
          "domainMin": -10,
          "domainMax": 10,
          "range": "width"
        },
        {
          "name": "y",
          "type": "linear",
          "round": true,
          "nice": true,
          "zero": true,
          "domainMin": -10,
          "domainMax": 10,
          "domain": {"data": "source", "field": "y"},
          "range": "height"
        }
      ],
      "axes": [
        {
          "scale": "x",
          "grid": false,
          "domain": false,
          "labels": false,
          "ticks": false,
          "offset": 0,
          "orient": "bottom"
        },
        {
          "scale": "y",
          "grid": false,
          "domain": false,
          "labels": false,
          "ticks": false,
          "offset": 0,
          "orient": "left"
        }
      ],
      "legends": [
      ],
      "marks": [
        {
          "name": "points",
          "clip": true,
          "type": "symbol",
          "from": {"data": "source"},
          "interactive": true,
          "hover": {},
          "encode": {
            "enter": {
              "x": {"scale": "x", "field": "x"},
              "y": {"scale": "y", "field": "y"},
              "shape": {"value": "circle"},
              "strokeWidth": {"value": 1},
              "size": {"signal": "width / 10"},
              "opacity": {"value": 0.8},
              "stroke": {"value": "black"},
              "fill": {"signal": "datum['color']"},
              // "href": {"signal": "'/gist/synspaces/' + datum.word"},
              "tooltip": {"signal": "{'Word': datum.word, 'Similarity': datum.similarity}"},
              "cursor": {"value": "pointer"}
            },
            "update": {
              "x": {"scale": "x", "field": "x"},
              "y": {"scale": "y", "field": "y"},
              "shape": {"value": "circle"},
              "strokeWidth": {"value": 1},
              "size": {"signal": "width / 10"},
              "opacity": {"value": 0.8},
              "stroke": {"value": "black"},
              "fill": {"signal": "datum['color']"}
            }
          }
        },
        { 
          "name": "words",
          "type": "text",
          "clip": true,
          "from": {"data": "points"},
          "encode": {
            "enter": {
              "text": {"field": "datum.word"},
              "fontSize": {"signal": "width / 35"},
            },
            "update": {
              "text": {"field": "datum.word"},
              "fontSize": {"signal": "width / 35"}
            },
          },
          "transform": [
            {
              "type": "label",
              "avoidMarks": ["points"],
              "offset": [1],
              "size": {"signal": "[width, height]"}
            }
          ]
        }
      ],
      "config": {}
    }

    return spec;
}

var app = new Vue({
  el: '#app',
  delimiters: ["[[", "]]"],
  data: {
    // Settings
    parameters:
      {
        embeddings_keys: embeddings_keys
      },
    status: "ready",
    // Selected Parameters
    embeddings_key: "glove_840B_wordnet",
    query: "",
    queries: [],
    n: 20,
    dimred: "pca",
    metric: "cosine",

    // Results
    results: []
    
  },
  mounted() {
    // this.get_synset('entity.n.01');
  },
  methods: {

    searchLemma: function() {
        let queries = this.queries;
        if (this.query && !this.queries.includes(this.query)) {
          queries.push(this.query)
          this.query = "";
        }
        var queryString = '/gist/synspaces_search?queries[]=' + queries.join(",");
        queryString += '&embeddings_key=' + this.embeddings_key;
        queryString += '&n=' + this.n;
        queryString += '&dimred=' + this.dimred;
        queryString += '&metric=' + this.metric;
        console.log(queryString)
        this.status="searching";
        axios.get(queryString).then(response => {
            console.log(response);
            this.results = response.data;
            this.drawVega(this.results);
            this.status="ready";
        });
    },

    addQuery: function() {
      if (this.query) {
        this.queries.push(this.query);
        this.query = "";
      }
    },

    clearQuery: function() {
      this.queries = [];
      this.query = "";
    },

    removeQuery: function(idx) {
      this.queries.splice(idx, 1)
      if (this.queries.length > 0) {
        this.searchLemma();
      }
    },

    drawVega(data) {
        const spec = buildVegaSpec(data);
        vegaEmbed("#results", spec, {"actions": false})
          // result.view provides access to the Vega View API
          .then(result => {
              console.log(result)
              result.view.addEventListener('click', function(event, item) {
                if (item && item.datum && item.datum.word) {
                  // console.log(event);
                  // console.log(item);
                  event.preventDefault();
                  // Remove tooltip
                  console.log(event.target)
                  simulate(event.target, "mouseout");
                  // oEvent.initEvent("mouseleave");
                  app.queries.push(item.datum.word);
                  app.searchLemma();
                }
              });})
          .catch(console.warn);
        },

  }
});