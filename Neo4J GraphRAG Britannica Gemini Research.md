# **Technical Specification: Neo4j GraphRAG for Encyclopædia Britannica (1768–1860)**

## **1\. Project Context & Objective**

Goal: Construct a Knowledge Graph from the first eight editions of Encyclopædia Britannica.  
Core Challenge: Transform unstructured OCR text into a structured property graph using the "New Plan" structure (Treatise vs. Article) and Heritage Textual Ontology (HTO).  
Target Database: Neo4j.

## **2\. Ontology & Graph Schema (HTO Alignment)**

The graph must align with the **Heritage Textual Ontology (HTO)**. Use the pre-computed EB\_Composite dataset (hto\_eb.ttl) as the structural backbone.

### **2.1 Node Labels & Properties**

| Neo4j Label | HTO Mapping | Description | Key Properties |
| :---- | :---- | :---- | :---- |
| **Concept** | hto:Work | Abstract intellectual content (e.g., "Chemistry"). | name, classification (Ed 8 ontology) |
| **Expression** | hto:Expression | Specific version of a concept in an edition (e.g., "Chemistry in 3rd Ed"). | edition\_id, text\_length |
| **Volume** | hto:Manifestation | Physical container. | vol\_number, pub\_year |
| **Treatise** | hto:Article | **Super Node**. Extensive "textbook" entry. | title, page\_start, page\_end, is\_treatise: true |
| **Article** | hto:Article | Standard dictionary definition or short entry. | title, term, is\_treatise: false |
| **Topic** | hto:Topic | Subject matter extracted/linked. | dbpedia\_id, wikidata\_id |
| **Person** | foaf:Person | Biographical entry or Author. | name, birth\_year, death\_year |
| **Contributor** | prov:Agent | Verified author of signed articles (Ed 3+). | name, specialty |

### **2.2 Relationship Types (Edge Logic)**

| Relationship | Source | Target | Logic / Heuristic |
| :---- | :---- | :---- | :---- |
| :APPEARS\_IN | Expression | Volume | Structural hierarchy. |
| :PART\_OF\_TREATISE | Section | Treatise | Hierarchical breakdown of long Treatises. |
| :DEFINES\_TERM | Article | Concept | Link specific text to abstract concept. |
| :SEE\_ALSO | Article | Treatise | Derived from "See X" regex. |
| :MENTIONS | Article | Entity | Derived from "q.v." or NER. |
| :SUPERSEDES | Supplement | Expression | **Critical:** Use for 1801/1815 Supplements correcting main text. |
| :AUTHORED\_BY | Treatise | Contributor | Only valid for signed articles (Ed 3 onwards). |

## **3\. Parsing Logic & Heuristics by Edition**

### **3.1 Structural Parsing (The "New Plan")**

* **Treatise Detection (Super Nodes):**  
  * *Visual Marker:* Large crossheads (centered caps) vs. inline bold headers.  
  * *Length Heuristic:* \>15 pages (Ed 1), \>50 pages (Ed 3+).  
  * *Content Structure:* Look for nested hierarchy: Treatise \-\> Part \-\> Chap \-\> Sect.  
* **Article Detection:**  
  * Inline headers. Short definitions.  
  * Must link to Treatises via :SEE\_ALSO edges.

### **3.2 Edition-Specific Handling**

#### **Edition 1 (1768–1771)**

* **Scope:** Arts & Sciences only. **Constraint:** No Person or History nodes allowed.  
* **Errata:** Vol 3 has erratic pagination. Rely on structural sequence over page numbers.  
* **Missing Assets:** Ignore references to "Plate CXII" (Midwifery) if image lookup fails (often censored/removed).

#### **Edition 2 (1777–1784)**

* **Scope Expansion:** Biography & History added. Person nodes valid.  
* **The Appendix Patch (Vol 10):**  
  * *Action:* Parse Vol 10 Appendix as a distinct sequence.  
  * *Logic:* Nodes in Appendix must have :SUPPLEMENTS edge to main graph nodes.

#### **Edition 3 (1788–1797) & Supplement (1801)**

* **Expert Authors:** Enable :AUTHORED\_BY extraction for specific sections (Robison, etc.).  
* **Supplement Logic:** Nodes in 1801 Supplement (2 Vols) often contradict main text (e.g., Chemistry/Phlogiston). Create :CORRECTS edge, do not merge text.

#### **Edition 4, 5, 6 (The Reprint Lineage)**

* **OCR Strategy:** **Ingest Edition 6 (1823) first.**  
  * *Reason:* Ed 6 abandoned the "long s" (ſ). It is the cleanest source text for the intellectual content of Eds 4-6.  
  * *Mapping:* Use Ed 6 text as the full\_text property for Ed 4/5 nodes if distinct scans are noisy.

#### **Edition 7 (1830–1842) \- The "Rosetta Stone"**

* **Priority Action:** Parse **Volume 22 (General Index)** immediately.  
* **Usage:** Use the Index to generate the canonical taxonomy of terms.  
  * *Pattern:* Term, sub-topic. See MainHeading. \-\> Creates (:Term)-\[:PART\_OF\]-\>(:Treatise).

## **4\. Text Engineering & Regex Targets**

### **4.1 Cross-Reference Extraction Patterns**

| Type | Pattern | Cypher Action | Note |
| :---- | :---- | :---- | :---- |
| **Direct** | \`(See | Vid.)\\s?(\[A-Z\]{4,})\` | MATCH (a:Article), (t:Treatise) CREATE (a)-\[:SEE\_ALSO\]-\>(t) |
| **Inline** | \`(\[A-Za-z\]+)\\s?((q.v. | which see))\` | MERGE (n:Entity) CREATE (current)-\[:MENTIONS\]-\>(n) |
| **Index** | ^(\[A-Z\]\[a-z\]+),\\s(.\*?)\\.\\sSee\\s(\[A-Z\]\[a-z\]+)\\.$ | CREATE (:Term {name: $1})-\[:INDEXED\_UNDER\]-\>(:Treatise {name: $3}) | Valid only for Vol 22 (Ed 7\) parsing. |

### **4.2 OCR Cleaning**

* **Long S (ſ):** s/f confusion.  
  * *Rule:* If Ed \< 6, apply heuristic: fight \-\> sight contextually.  
  * *Override:* Prefer Ed 6 text corpus where content overlaps.  
* **Hyphenation:**  
  * Use ALTO XML coordinates. If word ends in \- at x\_max, merge with line start of y+1.

## **5\. Data Source Mapping**

| Dataset Component | Source | Usage / Function |
| :---- | :---- | :---- |
| **Ontology/Skeleton** | EB\_Composite (Zenodo) | **Ingest First.** Provides hto\_eb.ttl RDF triples to build node structure. |
| **High-Fidelity Text** | 19KP TEI XML | Use for **Ed 3** and **Ed 7**. \>99% accuracy. Gold standard for training vectors. |
| **Broad Coverage** | NLS ALTO XML | Use for **Eds 1, 2, 4, 5, 6, 8**. Required for layout analysis (detecting Treatises by font size). |
| **Ground Truth Lists** | See Appendix A | Use for validating Contributor and Treatise classifications. |

## **6\. Known "Super Nodes" (Persistent Treatises)**

*Ensure special handling/indexing for these persistent domains:*

* Anatomy, Chemistry, Medicine, Optics, Astronomy, Law.  
* *Note:* History and Biography are NOT Treatises in Ed 1, but ARE in Ed 2+.

## **7\. Contributor Entity Extraction (Supplement to 4-6)**

*Regex targeting signatures/attribution for known authors:*

* **Thomas Young:** Egypt, Chromatics, Tides  
* **James Mill:** Government, Jurisprudence  
* **David Ricardo:** Funding System  
* **Walter Scott:** Chivalry, Romance