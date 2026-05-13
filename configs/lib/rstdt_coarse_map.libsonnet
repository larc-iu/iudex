// Carlson & Marcu (2001) fine→coarse mapping for RST-DT, folding the ~110
// fine-grained labels into the 18 coarse classes used by DMRST, Yu et al.,
// and most RST-DT parsing benchmarks.
//
// Naming: coarse class names are lowercase-hyphenated. Mononuclear fine
// variants (suffixes -e for embedded, -n for nucleus side, -s for
// satellite side) all fold to the same coarse class as their base form.
// The structural "span" marker is handled by the reader and need not
// appear here.
{
    // attribution
    "attribution": "attribution",
    "attribution-e": "attribution",
    "attribution-n": "attribution",

    // background
    "background": "background",
    "background-e": "background",
    "circumstance": "background",
    "circumstance-e": "background",

    // cause
    "cause": "cause",
    "consequence-n": "cause",
    "consequence-n-e": "cause",
    "consequence-s": "cause",
    "consequence-s-e": "cause",
    "result": "cause",
    "result-e": "cause",
    "Cause-Result": "cause",
    "Consequence": "cause",

    // comparison
    "analogy": "comparison",
    "analogy-e": "comparison",
    "comparison": "comparison",
    "comparison-e": "comparison",
    "preference": "comparison",
    "preference-e": "comparison",
    "Analogy": "comparison",
    "Comparison": "comparison",
    "Proportion": "comparison",

    // condition
    "condition": "condition",
    "condition-e": "condition",
    "contingency": "condition",
    "hypothetical": "condition",
    "otherwise": "condition",
    "Otherwise": "condition",

    // contrast
    "antithesis": "contrast",
    "antithesis-e": "contrast",
    "concession": "contrast",
    "concession-e": "contrast",
    "Contrast": "contrast",

    // elaboration
    "definition": "elaboration",
    "definition-e": "elaboration",
    "elaboration-additional": "elaboration",
    "elaboration-additional-e": "elaboration",
    "elaboration-general-specific": "elaboration",
    "elaboration-general-specific-e": "elaboration",
    "elaboration-object-attribute": "elaboration",
    "elaboration-object-attribute-e": "elaboration",
    "elaboration-part-whole": "elaboration",
    "elaboration-part-whole-e": "elaboration",
    "elaboration-process-step": "elaboration",
    "elaboration-process-step-e": "elaboration",
    "elaboration-set-member": "elaboration",
    "elaboration-set-member-e": "elaboration",
    "example": "elaboration",
    "example-e": "elaboration",

    // enablement
    "enablement": "enablement",
    "enablement-e": "enablement",
    "purpose": "enablement",
    "purpose-e": "enablement",

    // evaluation
    "comment": "evaluation",
    "comment-e": "evaluation",
    "conclusion": "evaluation",
    "evaluation-n": "evaluation",
    "evaluation-s": "evaluation",
    "evaluation-s-e": "evaluation",
    "interpretation-n": "evaluation",
    "interpretation-s": "evaluation",
    "interpretation-s-e": "evaluation",
    "Evaluation": "evaluation",
    "Interpretation": "evaluation",

    // explanation
    "evidence": "explanation",
    "evidence-e": "explanation",
    "explanation-argumentative": "explanation",
    "explanation-argumentative-e": "explanation",
    "reason": "explanation",
    "reason-e": "explanation",
    "Reason": "explanation",

    // joint
    "List": "joint",
    "Disjunction": "joint",

    // manner-means
    "manner": "manner-means",
    "manner-e": "manner-means",
    "means": "manner-means",
    "means-e": "manner-means",

    // summary
    "restatement": "summary",
    "restatement-e": "summary",
    "summary-n": "summary",
    "summary-s": "summary",

    // temporal
    "temporal-after": "temporal",
    "temporal-after-e": "temporal",
    "temporal-before": "temporal",
    "temporal-before-e": "temporal",
    "temporal-same-time": "temporal",
    "temporal-same-time-e": "temporal",
    "Sequence": "temporal",
    "Inverted-Sequence": "temporal",
    "Temporal-Same-Time": "temporal",

    // topic-comment
    "problem-solution-n": "topic-comment",
    "problem-solution-s": "topic-comment",
    "question-answer-n": "topic-comment",
    "question-answer-s": "topic-comment",
    "rhetorical-question": "topic-comment",
    "statement-response-n": "topic-comment",
    "statement-response-s": "topic-comment",
    "Problem-Solution": "topic-comment",
    "Question-Answer": "topic-comment",
    "Statement-Response": "topic-comment",
    "Topic-Comment": "topic-comment",
    "Comment-Topic": "topic-comment",

    // topic-change
    "topic-drift": "topic-change",
    "topic-shift": "topic-change",
    "Topic-Drift": "topic-change",
    "Topic-Shift": "topic-change",

    // same-unit
    "Same-Unit": "same-unit",

    // textual-organization
    "TextualOrganization": "textual-organization",
}
