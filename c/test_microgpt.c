/**
 * Comprehensive test suite for microgpt C implementation
 * Compile: gcc -O3 -o test_microgpt test_microgpt.c microgpt.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define EPSILON 1e-9

// External declarations from microgpt.c
typedef struct {
    uint32_t mt[624];
    int mti;
    double gauss_next;
    int has_gauss_next;
} MersenneTwister;

extern void mt_init(MersenneTwister *rng, uint32_t seed);
extern uint32_t mt_genrand_int32(MersenneTwister *rng);
extern double mt_random(MersenneTwister *rng);
extern double mt_gauss(MersenneTwister *rng, double mu, double sigma);
extern void mt_shuffle(MersenneTwister *rng, int *arr, int n);
extern int mt_choices(MersenneTwister *rng, int *population, double *weights, int n);

extern void value_pool_reset(void);
extern int value_new(double data);
extern int value_add(int a, int b);
extern int value_mul(int a, int b);
extern int value_sub(int a, int b);
extern int value_div(int a, int b);
extern int value_pow(int a, double n);
extern int value_neg(int a);
extern int value_log(int a);
extern int value_exp(int a);
extern int value_relu(int a);
extern void value_backward(int v);
extern double value_get_data(int idx);
extern double value_get_grad(int idx);

extern void softmax(double *logits, double *probs, int n);
extern void rmsnorm(double *x, double *out, int n);
extern void linear(double *x, double **w, double *out, int nout, int nin);

// Reference values
const double PRNG_RANDOM_100[100] = {
    0.6394267984578837, 0.025010755222666936, 0.27502931836911926, 0.22321073814882275,
    0.7364712141640124, 0.6766994874229113, 0.8921795677048454, 0.08693883262941615,
    0.4219218196852704, 0.029797219438070344, 0.21863797480360336, 0.5053552881033624,
    0.026535969683863625, 0.1988376506866485, 0.6498844377795232, 0.5449414806032167,
    0.2204406220406967, 0.5892656838759087, 0.8094304566778266, 0.006498759678061017,
    0.8058192518328079, 0.6981393949882269, 0.3402505165179919, 0.15547949981178155,
    0.9572130722067812, 0.33659454511262676, 0.09274584338014791, 0.09671637683346401,
    0.8474943663474598, 0.6037260313668911, 0.8071282732743802, 0.7297317866938179,
    0.5362280914547007, 0.9731157639793706, 0.3785343772083535, 0.552040631273227,
    0.8294046642529949, 0.6185197523642461, 0.8617069003107772, 0.577352145256762,
    0.7045718362149235, 0.045824383655662215, 0.22789827565154686, 0.28938796360210717,
    0.0797919769236275, 0.23279088636103018, 0.10100142940972912, 0.2779736031100921,
    0.6356844442644002, 0.36483217897008424, 0.37018096711688264, 0.2095070307714877,
    0.26697782204911336, 0.936654587712494, 0.6480353852465935, 0.6091310056669882,
    0.171138648198097, 0.7291267979503492, 0.1634024937619284, 0.3794554417576478,
    0.9895233506365952, 0.6399997598540929, 0.5569497437746462, 0.6846142509898746,
    0.8428519201898096, 0.7759999115462448, 0.22904807196410437, 0.03210024390403776,
    0.3154530480590819, 0.26774087597570273, 0.21098284358632646, 0.9429097143350544,
    0.8763676264726689, 0.3146778807984779, 0.65543866529488, 0.39563190106066426,
    0.9145475897405435, 0.4588518525873988, 0.26488016649805246, 0.24662750769398345,
    0.5613681341631508, 0.26274160852293527, 0.5845859902235405, 0.897822883602477,
    0.39940050514039727, 0.21932075915728333, 0.9975376064951103, 0.5095262936764645,
    0.09090941217379389, 0.04711637542473457, 0.10964913035065915, 0.62744604170309,
    0.7920793643629641, 0.42215996679968404, 0.06352770615195713, 0.38161928650653676,
    0.9961213802400968, 0.529114345099137, 0.9710783776136181, 0.8607797022344981
};

const int PRNG_SHUFFLE_100[100] = {
    42, 41, 91, 9, 65, 50, 1, 70, 15, 78, 73, 10, 55, 56, 72, 45, 48, 92, 76, 37,
    30, 21, 32, 96, 80, 49, 83, 26, 87, 33, 8, 47, 59, 63, 74, 44, 98, 52, 85, 12,
    36, 23, 39, 40, 18, 66, 61, 60, 7, 34, 99, 46, 2, 51, 16, 38, 58, 68, 22, 62,
    24, 5, 6, 67, 82, 19, 79, 43, 90, 20, 0, 95, 57, 93, 53, 89, 25, 71, 84, 77,
    64, 29, 27, 88, 97, 4, 54, 75, 11, 69, 86, 13, 17, 28, 31, 35, 94, 3, 14, 81
};

const int PRNG_CHOICES_100[100] = {
    3, 0, 1, 1, 3, 3, 3, 0, 2, 0, 1, 2, 0, 1, 3, 2, 1, 2, 3, 0, 3, 3, 2, 1, 3, 2, 0, 0, 3,
    3, 3, 3, 2, 3, 2, 2, 3, 3, 3, 2, 3, 0, 1, 1, 0, 1, 1, 1, 3, 2, 2, 1, 1, 3, 3, 3, 1, 3,
    1, 2, 3, 3, 2, 3, 3, 3, 1, 0, 2, 1, 1, 3, 3, 2, 3, 2, 3, 2, 1, 1, 2, 1, 2, 3, 2, 1, 3,
    2, 0, 0, 1, 3, 3, 2, 0, 2, 3, 2, 3, 3
};

int approx_eq(double a, double b, double eps) {
    return fabs(a - b) < eps;
}

// ============================================================================
// PRNG Tests
// ============================================================================

void test_prng_random_matches_python() {
    printf("Testing PRNG random() matches Python...\n");
    MersenneTwister rng;
    mt_init(&rng, 42);
    
    int passed = 1;
    for (int i = 0; i < 100; i++) {
        double actual = mt_random(&rng);
        if (!approx_eq(actual, PRNG_RANDOM_100[i], EPSILON)) {
            printf("  FAIL: random()[%d]: expected %.17f, got %.17f\n", 
                   i, PRNG_RANDOM_100[i], actual);
            passed = 0;
            if (i >= 5) break;
        }
    }
    
    if (passed) {
        printf("  PASS: All 100 random() values match Python\n");
    }
}

void test_prng_random_values_in_range() {
    printf("Testing PRNG random() values in range [0, 1)...\n");
    MersenneTwister rng;
    mt_init(&rng, 42);
    
    int passed = 1;
    for (int i = 0; i < 100; i++) {
        double val = mt_random(&rng);
        if (val < 0.0 || val >= 1.0) {
            printf("  FAIL: Value %f out of range at %d\n", val, i);
            passed = 0;
            break;
        }
    }
    
    if (passed) {
        printf("  PASS: All values in range [0, 1)\n");
    }
}

void test_prng_random_deterministic() {
    printf("Testing PRNG random() is deterministic...\n");
    MersenneTwister rng1, rng2;
    mt_init(&rng1, 42);
    mt_init(&rng2, 42);
    
    int passed = 1;
    for (int i = 0; i < 100; i++) {
        if (mt_random(&rng1) != mt_random(&rng2)) {
            printf("  FAIL: Different values at %d\n", i);
            passed = 0;
            break;
        }
    }
    
    if (passed) {
        printf("  PASS: Same seed produces same sequence\n");
    }
}

void test_prng_random_different_seeds() {
    printf("Testing PRNG random() different seeds...\n");
    MersenneTwister rng1, rng2;
    mt_init(&rng1, 42);
    mt_init(&rng2, 123);
    
    int same = 0;
    for (int i = 0; i < 100; i++) {
        if (mt_random(&rng1) == mt_random(&rng2)) same++;
    }
    
    if (same < 5) {
        printf("  PASS: Different seeds produce different sequences (%d matches)\n", same);
    } else {
        printf("  FAIL: Too many matches: %d\n", same);
    }
}

void test_prng_gauss_distribution() {
    printf("Testing PRNG gauss() distribution...\n");
    MersenneTwister rng;
    mt_init(&rng, 42);
    
    double sum = 0.0;
    for (int i = 0; i < 1000; i++) {
        sum += mt_gauss(&rng, 0.0, 1.0);
    }
    double mean = sum / 1000.0;
    
    if (fabs(mean) < 0.1) {
        printf("  PASS: gauss() mean ~0 (got %f)\n", mean);
    } else {
        printf("  FAIL: gauss() mean should be ~0, got %f\n", mean);
    }
}

void test_prng_shuffle_matches_python() {
    printf("Testing PRNG shuffle() matches Python...\n");
    MersenneTwister rng;
    mt_init(&rng, 42);
    
    int arr[100];
    for (int i = 0; i < 100; i++) arr[i] = i;
    
    mt_shuffle(&rng, arr, 100);
    
    int passed = 1;
    for (int i = 0; i < 100; i++) {
        if (arr[i] != PRNG_SHUFFLE_100[i]) {
            printf("  FAIL: shuffle[%d]: expected %d, got %d\n", i, PRNG_SHUFFLE_100[i], arr[i]);
            passed = 0;
            if (i >= 5) break;
        }
    }
    
    if (passed) {
        printf("  PASS: All 100 shuffle() values match Python\n");
    }
}

void test_prng_shuffle_preserves_elements() {
    printf("Testing PRNG shuffle() preserves elements...\n");
    MersenneTwister rng;
    mt_init(&rng, 42);
    
    int arr[100];
    for (int i = 0; i < 100; i++) arr[i] = i;
    
    mt_shuffle(&rng, arr, 100);
    
    int sorted[100];
    memcpy(sorted, arr, sizeof(arr));
    for (int i = 0; i < 99; i++) {
        for (int j = i + 1; j < 100; j++) {
            if (sorted[i] > sorted[j]) {
                int t = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = t;
            }
        }
    }
    
    int passed = 1;
    for (int i = 0; i < 100; i++) {
        if (sorted[i] != i) {
            passed = 0;
            break;
        }
    }
    
    if (passed) {
        printf("  PASS: shuffle() preserves all elements\n");
    } else {
        printf("  FAIL: Elements were lost\n");
    }
}

void test_prng_choices_matches_python() {
    printf("Testing PRNG choices() matches Python...\n");
    MersenneTwister rng;
    mt_init(&rng, 42);
    
    int population[] = {0, 1, 2, 3};
    double weights[] = {1.0, 2.0, 3.0, 4.0};
    
    int passed = 1;
    for (int i = 0; i < 100; i++) {
        int actual = mt_choices(&rng, population, weights, 4);
        if (actual != PRNG_CHOICES_100[i]) {
            printf("  FAIL: choices[%d]: expected %d, got %d\n", i, PRNG_CHOICES_100[i], actual);
            passed = 0;
            if (i >= 5) break;
        }
    }
    
    if (passed) {
        printf("  PASS: All 100 choices() values match Python\n");
    }
}

// ============================================================================
// Value Tests
// ============================================================================

void test_value_add() {
    printf("Testing Value add...\n");
    value_pool_reset();
    int a = value_new(2.0);
    int b = value_new(3.0);
    int c = value_add(a, b);
    
    if (approx_eq(value_get_data(c), 5.0, EPSILON)) {
        printf("  PASS: 2 + 3 = 5\n");
    } else {
        printf("  FAIL: expected 5, got %f\n", value_get_data(c));
    }
}

void test_value_add_gradient() {
    printf("Testing Value add gradient...\n");
    value_pool_reset();
    int a = value_new(2.0);
    int b = value_new(3.0);
    int c = value_add(a, b);
    
    value_backward(c);
    
    if (approx_eq(value_get_grad(a), 1.0, EPSILON) && approx_eq(value_get_grad(b), 1.0, EPSILON)) {
        printf("  PASS: gradients are 1\n");
    } else {
        printf("  FAIL: gradients should be 1, got %f and %f\n", value_get_grad(a), value_get_grad(b));
    }
}

void test_value_mul() {
    printf("Testing Value mul...\n");
    value_pool_reset();
    int a = value_new(2.0);
    int b = value_new(3.0);
    int c = value_mul(a, b);
    
    if (approx_eq(value_get_data(c), 6.0, EPSILON)) {
        printf("  PASS: 2 * 3 = 6\n");
    } else {
        printf("  FAIL: expected 6, got %f\n", value_get_data(c));
    }
}

void test_value_mul_gradient() {
    printf("Testing Value mul gradient...\n");
    value_pool_reset();
    int a = value_new(2.0);
    int b = value_new(3.0);
    int c = value_mul(a, b);
    
    value_backward(c);
    
    if (approx_eq(value_get_grad(a), 3.0, EPSILON) && approx_eq(value_get_grad(b), 2.0, EPSILON)) {
        printf("  PASS: gradients are y=3 and x=2\n");
    } else {
        printf("  FAIL: gradients should be 3 and 2, got %f and %f\n", value_get_grad(a), value_get_grad(b));
    }
}

void test_value_pow() {
    printf("Testing Value pow...\n");
    value_pool_reset();
    int a = value_new(2.0);
    int c = value_pow(a, 3.0);
    
    if (approx_eq(value_get_data(c), 8.0, EPSILON)) {
        printf("  PASS: 2^3 = 8\n");
    } else {
        printf("  FAIL: expected 8, got %f\n", value_get_data(c));
    }
}

void test_value_sub() {
    printf("Testing Value sub...\n");
    value_pool_reset();
    int a = value_new(5.0);
    int b = value_new(3.0);
    int c = value_sub(a, b);
    
    if (approx_eq(value_get_data(c), 2.0, EPSILON)) {
        printf("  PASS: 5 - 3 = 2\n");
    } else {
        printf("  FAIL: expected 2, got %f\n", value_get_data(c));
    }
}

void test_value_div() {
    printf("Testing Value div...\n");
    value_pool_reset();
    int a = value_new(6.0);
    int b = value_new(2.0);
    int c = value_div(a, b);
    
    if (approx_eq(value_get_data(c), 3.0, EPSILON)) {
        printf("  PASS: 6 / 2 = 3\n");
    } else {
        printf("  FAIL: expected 3, got %f\n", value_get_data(c));
    }
}

void test_value_neg() {
    printf("Testing Value neg...\n");
    value_pool_reset();
    int a = value_new(5.0);
    int c = value_neg(a);
    
    if (approx_eq(value_get_data(c), -5.0, EPSILON)) {
        printf("  PASS: -5 = -5\n");
    } else {
        printf("  FAIL: expected -5, got %f\n", value_get_data(c));
    }
}

void test_value_log() {
    printf("Testing Value log...\n");
    value_pool_reset();
    int a = value_new(2.718281828459045);
    int c = value_log(a);
    
    if (approx_eq(value_get_data(c), 1.0, EPSILON)) {
        printf("  PASS: log(e) = 1\n");
    } else {
        printf("  FAIL: expected 1, got %f\n", value_get_data(c));
    }
}

void test_value_exp() {
    printf("Testing Value exp...\n");
    value_pool_reset();
    int a = value_new(1.0);
    int c = value_exp(a);
    
    if (approx_eq(value_get_data(c), 2.718281828459045, EPSILON)) {
        printf("  PASS: exp(1) = e\n");
    } else {
        printf("  FAIL: expected e, got %f\n", value_get_data(c));
    }
}

void test_value_relu_positive() {
    printf("Testing Value relu positive...\n");
    value_pool_reset();
    int a = value_new(5.0);
    int c = value_relu(a);
    
    if (approx_eq(value_get_data(c), 5.0, EPSILON)) {
        printf("  PASS: relu(5) = 5\n");
    } else {
        printf("  FAIL: expected 5, got %f\n", value_get_data(c));
    }
}

void test_value_relu_negative() {
    printf("Testing Value relu negative...\n");
    value_pool_reset();
    int a = value_new(-5.0);
    int c = value_relu(a);
    
    if (approx_eq(value_get_data(c), 0.0, EPSILON)) {
        printf("  PASS: relu(-5) = 0\n");
    } else {
        printf("  FAIL: expected 0, got %f\n", value_get_data(c));
    }
}

// ============================================================================
// Neural Network Tests
// ============================================================================

void test_softmax() {
    printf("Testing softmax...\n");
    double logits[3] = {1.0, 2.0, 3.0};
    double probs[3];
    softmax(logits, probs, 3);
    
    double ref[3] = {0.09003057317038045, 0.2447284710547976, 0.6652409557748218};
    
    int passed = 1;
    for (int i = 0; i < 3; i++) {
        if (!approx_eq(probs[i], ref[i], 1e-9)) {
            printf("  FAIL: softmax[%d]: expected %.17f, got %.17f\n", i, ref[i], probs[i]);
            passed = 0;
        }
    }
    
    if (passed) {
        printf("  PASS: softmax values match Python\n");
    }
}

void test_softmax_sums_to_one() {
    printf("Testing softmax sums to 1...\n");
    double logits[5] = {1.0, 2.0, 3.0, 4.0, 5.0};
    double probs[5];
    softmax(logits, probs, 5);
    
    double sum = 0.0;
    for (int i = 0; i < 5; i++) sum += probs[i];
    
    if (approx_eq(sum, 1.0, 1e-9)) {
        printf("  PASS: softmax sums to 1\n");
    } else {
        printf("  FAIL: softmax sums to %f\n", sum);
    }
}

void test_softmax_numerical_stability() {
    printf("Testing softmax numerical stability...\n");
    double logits[3] = {1000.0, 1001.0, 1002.0};
    double probs[3];
    softmax(logits, probs, 3);
    
    double ref[3] = {0.09003057317038045, 0.2447284710547976, 0.6652409557748218};
    
    int passed = 1;
    for (int i = 0; i < 3; i++) {
        if (!approx_eq(probs[i], ref[i], 1e-9)) {
            printf("  FAIL: softmax[%d]: expected %.17f, got %.17f\n", i, ref[i], probs[i]);
            passed = 0;
        }
    }
    
    if (passed) {
        printf("  PASS: softmax handles large values\n");
    }
}

void test_rmsnorm() {
    printf("Testing rmsnorm...\n");
    double x[3] = {1.0, 2.0, 3.0};
    double out[3];
    rmsnorm(x, out, 3);
    
    double ref[3] = {0.46290955391201943, 0.9258191078240389, 1.3887286617360584};
    
    int passed = 1;
    for (int i = 0; i < 3; i++) {
        if (!approx_eq(out[i], ref[i], 1e-9)) {
            printf("  FAIL: rmsnorm[%d]: expected %.17f, got %.17f\n", i, ref[i], out[i]);
            passed = 0;
        }
    }
    
    if (passed) {
        printf("  PASS: rmsnorm values match Python\n");
    }
}

void test_rmsnorm_same_values() {
    printf("Testing rmsnorm with same values...\n");
    double x[3] = {5.0, 5.0, 5.0};
    double out[3];
    rmsnorm(x, out, 3);
    
    int passed = 1;
    for (int i = 0; i < 3; i++) {
        if (!approx_eq(out[i], 1.0, 1e-5)) {
            printf("  FAIL: rmsnorm[%d]: expected ~1, got %f\n", i, out[i]);
            passed = 0;
        }
    }
    
    if (passed) {
        printf("  PASS: rmsnorm of same values is ~1\n");
    }
}

void test_linear() {
    printf("Testing linear...\n");
    double x[3] = {1.0, 2.0, 3.0};
    double *w[2];
    double w_data[2][3] = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    w[0] = w_data[0];
    w[1] = w_data[1];
    double out[2];
    
    linear(x, w, out, 2, 3);
    
    double ref[2] = {14.0, 32.0};
    
    int passed = 1;
    for (int i = 0; i < 2; i++) {
        if (!approx_eq(out[i], ref[i], EPSILON)) {
            printf("  FAIL: linear[%d]: expected %f, got %f\n", i, ref[i], out[i]);
            passed = 0;
        }
    }
    
    if (passed) {
        printf("  PASS: linear values match Python\n");
    }
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== microgpt C Test Suite ===\n\n");
    
    printf("--- PRNG Tests ---\n");
    test_prng_random_matches_python();
    test_prng_random_values_in_range();
    test_prng_random_deterministic();
    test_prng_random_different_seeds();
    test_prng_gauss_distribution();
    test_prng_shuffle_matches_python();
    test_prng_shuffle_preserves_elements();
    test_prng_choices_matches_python();
    
    printf("\n--- Value Tests ---\n");
    test_value_add();
    test_value_add_gradient();
    test_value_mul();
    test_value_mul_gradient();
    test_value_sub();
    test_value_div();
    test_value_pow();
    test_value_neg();
    test_value_log();
    test_value_exp();
    test_value_relu_positive();
    test_value_relu_negative();
    
    printf("\n--- Neural Network Tests ---\n");
    test_softmax();
    test_softmax_sums_to_one();
    test_softmax_numerical_stability();
    test_rmsnorm();
    test_rmsnorm_same_values();
    test_linear();
    
    printf("\n=== Test suite complete ===\n");
    return 0;
}
