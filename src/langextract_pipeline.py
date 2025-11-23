"""LangExtract-based approach for NER extraction from news articles."""

import sys
import textwrap
from typing import Dict, List
from collections import Counter
import langextract as lx
from tqdm import tqdm

sys.path.append(".")
from src.config import NERLangExtractConfig
from src.data_loader import NERDataLoader


class LangExtractNERExtractor:
    """NER extraction using Google's LangExtract library."""

    def __init__(self, config: NERLangExtractConfig = None, model_id: str = None):
        """
        Initialize LangExtract-based NER extractor.

        Args:
            config: Configuration object
            model_id: Gemini model ID to use (overrides config if provided)
        """

        self.config = config or NERLangExtractConfig()
        self.model_id = model_id or self.config.model_id

        # Define extraction prompt
        self.prompt = textwrap.dedent("""\
            Extract named entities from the given news article text.

            Identify and extract three types of entities:
            1. person - Names of people (individuals, politicians, celebrities, etc.)
            2. organizations - Names of companies, institutions, organizations, agencies
            3. address - Locations, places, addresses, geographical names (cities, countries, regions)

            Important guidelines:
            - Use exact text from the input for extraction_text. Do not paraphrase.
            - Extract entities in order of appearance with no overlapping text spans.
            - Provide meaningful attributes when possible (e.g., role, context, type).
            - Skip common words that are not actual entity names.
            - For organizations, include full official names when mentioned.
            - For locations, extract complete place names (e.g., "New York City" not just "New York").""")

        # Define extraction examples
        self.examples = [
            lx.data.ExampleData(
                text=textwrap.dedent("""\
                    Apple Inc. announced that CEO Tim Cook will speak at a technology conference
                    in San Francisco next week. The company, based in Cupertino, California,
                    plans to unveil new products. Microsoft and Google are also expected to attend."""),
                extractions=[
                    lx.data.Extraction(
                        extraction_class="organizations",
                        extraction_text="Apple Inc.",
                        attributes={"type": "technology company", "role": "main subject"}
                    ),
                    lx.data.Extraction(
                        extraction_class="person",
                        extraction_text="Tim Cook",
                        attributes={"role": "CEO of Apple", "context": "speaker"}
                    ),
                    lx.data.Extraction(
                        extraction_class="address",
                        extraction_text="San Francisco",
                        attributes={"type": "city", "context": "event location"}
                    ),
                    lx.data.Extraction(
                        extraction_class="address",
                        extraction_text="Cupertino",
                        attributes={"type": "city", "context": "company headquarters"}
                    ),
                    lx.data.Extraction(
                        extraction_class="address",
                        extraction_text="California",
                        attributes={"type": "state"}
                    ),
                    lx.data.Extraction(
                        extraction_class="organizations",
                        extraction_text="Microsoft",
                        attributes={"type": "technology company", "context": "attendee"}
                    ),
                    lx.data.Extraction(
                        extraction_class="organizations",
                        extraction_text="Google",
                        attributes={"type": "technology company", "context": "attendee"}
                    ),
                ]
            ),
            lx.data.ExampleData(
                text=textwrap.dedent("""\
                    President Joe Biden met with Prime Minister of the United Kingdom at the
                    White House in Washington D.C. yesterday. The European Union and NATO
                    representatives also participated in the discussions."""),
                extractions=[
                    lx.data.Extraction(
                        extraction_class="person",
                        extraction_text="Joe Biden",
                        attributes={"role": "President", "context": "meeting participant"}
                    ),
                    lx.data.Extraction(
                        extraction_class="person",
                        extraction_text="Prime Minister of the United Kingdom",
                        attributes={"role": "government leader", "context": "meeting participant"}
                    ),
                    lx.data.Extraction(
                        extraction_class="address",
                        extraction_text="White House",
                        attributes={"type": "building", "context": "meeting location"}
                    ),
                    lx.data.Extraction(
                        extraction_class="address",
                        extraction_text="Washington D.C.",
                        attributes={"type": "city", "context": "location"}
                    ),
                    lx.data.Extraction(
                        extraction_class="organizations",
                        extraction_text="European Union",
                        attributes={"type": "international organization", "context": "participant"}
                    ),
                    lx.data.Extraction(
                        extraction_class="organizations",
                        extraction_text="NATO",
                        attributes={"type": "international alliance", "context": "participant"}
                    ),
                ]
            ),
        ]

    def extract_entities(
        self,
        text: str,
        extraction_passes: int = 2,
        max_workers: int = 10,
        max_char_buffer: int = 2000
    ) -> Dict[str, List[str]]:
        """
        Extract entities from text using LangExtract.

        Args:
            text: Input text
            extraction_passes: Number of extraction passes for better recall
            max_workers: Number of parallel workers
            max_char_buffer: Maximum character buffer size for chunking

        Returns:
            Dictionary with extracted entities
        """
        text = NERDataLoader.preprocess_text(text)

        result = lx.extract(
            model_id=self.model_id,
            api_key=self.config.api_key,
            text_or_documents=text,
            prompt_description=self.prompt,
            examples=self.examples,
            extraction_passes=extraction_passes,
            max_workers=max_workers,
            max_char_buffer=max_char_buffer,
            debug=False
        )

        entities = self._convert_to_ner_format(result)

        return entities

    def _convert_to_ner_format(self, result) -> Dict[str, List[str]]:
        """
        Convert LangExtract result to NER format.

        Args:
            result: LangExtract result object

        Returns:
            Dictionary with entity lists
        """
        entities = {"person": [], "organizations": [], "address": []}

        # Group extractions by class
        for extraction in result.extractions:
            entity_class = extraction.extraction_class
            entity_text = extraction.extraction_text.strip()

            # Map to our entity types
            if entity_class in entities and entity_text:
                # Avoid duplicates
                if entity_text not in entities[entity_class]:
                    entities[entity_class].append(entity_text)

        return entities

    def extract_with_details(
        self,
        text: str,
        extraction_passes: int = 2,
        max_workers: int = 10,
        max_char_buffer: int = 2000
    ) -> Dict:
        """
        Extract entities with detailed attributes.

        Args:
            text: Input text
            extraction_passes: Number of extraction passes
            max_workers: Number of parallel workers
            max_char_buffer: Maximum character buffer size

        Returns:
            Dictionary with detailed extraction results
        """
        # Preprocess text
        text = NERDataLoader.preprocess_text(text)

        # Run LangExtract
        result = lx.extract(
            text_or_documents=text,
            prompt_description=self.prompt,
            examples=self.examples,
            model_id=self.model_id,
            extraction_passes=extraction_passes,
            max_workers=max_workers,
            max_char_buffer=max_char_buffer
        )

        # Convert to detailed format
        detailed_entities = {
            "person": [],
            "organizations": [],
            "address": [],
            "statistics": {
                "total_extractions": len(result.extractions),
                "text_length": len(result.text),
                "extraction_passes": extraction_passes
            }
        }

        # Group extractions with details
        for extraction in result.extractions:
            entity_class = extraction.extraction_class
            if entity_class in detailed_entities:
                entity_info = {
                    "text": extraction.extraction_text.strip(),
                    "attributes": extraction.attributes or {}
                }

                # Check for duplicates
                existing_texts = [e["text"] for e in detailed_entities[entity_class]]
                if entity_info["text"] and entity_info["text"] not in existing_texts:
                    detailed_entities[entity_class].append(entity_info)

        return detailed_entities

    def batch_extract(
        self,
        texts: List[str],
        show_progress: bool = True,
        extraction_passes: int = 2,
        max_workers: int = 10
    ) -> List[Dict[str, List[str]]]:
        """
        Extract entities from multiple texts.

        Args:
            texts: List of input texts
            show_progress: Whether to show progress bar
            extraction_passes: Number of extraction passes
            max_workers: Number of parallel workers

        Returns:
            List of entity dictionaries
        """
        results = []

        iterator = tqdm(texts, desc="Extracting entities (LangExtract)") if show_progress else texts

        for text in iterator:
            try:
                entities = self.extract_entities(
                    text,
                    extraction_passes=extraction_passes,
                    max_workers=max_workers
                )
                results.append(entities)
            except Exception as e:
                print(f"Warning: Extraction failed for text: {e}")
                # Return empty entities on error
                results.append({"person": [], "organizations": [], "address": []})

        return results

    def evaluate_on_dataset(self, dataset: List[Dict]) -> tuple:
        """
        Run extraction on a dataset.

        Args:
            dataset: List of samples with 'text' and 'entities' keys

        Returns:
            Tuple of (predictions, ground_truth)
        """
        texts = [sample["text"] for sample in dataset]
        predictions = self.batch_extract(texts)
        ground_truth = [sample["entities"] for sample in dataset]

        return predictions, ground_truth

    def save_annotated_documents(
        self,
        texts: List[str],
        output_path: str,
        extraction_passes: int = 2
    ):
        """
        Save annotated documents in JSONL format.

        Args:
            texts: List of texts to process
            output_path: Output file path
            extraction_passes: Number of extraction passes
        """
        results = []

        for text in tqdm(texts, desc="Processing documents"):
            result = lx.extract(
                text_or_documents=text,
                prompt_description=self.prompt,
                examples=self.examples,
                model_id=self.model_id,
                extraction_passes=extraction_passes
            )
            results.append(result)

        # Save to JSONL
        lx.io.save_annotated_documents(results, output_name=output_path)

        print(f"Saved {len(results)} annotated documents to {output_path}")

    def create_visualization(
        self,
        jsonl_path: str,
        output_html_path: str
    ):
        """
        Create interactive HTML visualization from JSONL results.

        Args:
            jsonl_path: Path to JSONL file with extraction results
            output_html_path: Path to save HTML visualization
        """
        html_content = lx.visualize(jsonl_path)

        with open(output_html_path, "w", encoding="utf-8") as f:
            if hasattr(html_content, 'data'):
                f.write(html_content.data)  # For Jupyter/Colab
            else:
                f.write(html_content)

        print(f"Interactive visualization saved to {output_html_path}")

    @staticmethod
    def analyze_extraction_statistics(jsonl_path: str) -> Dict:
        """
        Analyze extraction statistics from JSONL results.

        Args:
            jsonl_path: Path to JSONL file

        Returns:
            Dictionary with statistics
        """
        import json

        all_extractions = []
        total_chars = 0
        total_docs = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                total_docs += 1
                total_chars += len(doc.get("text", ""))

                for extraction in doc.get("extractions", []):
                    all_extractions.append(extraction)

        # Count by class
        class_counts = Counter(e["extraction_class"] for e in all_extractions)

        # Unique entities
        unique_entities = {}
        for extraction in all_extractions:
            entity_class = extraction["extraction_class"]
            entity_text = extraction["extraction_text"]

            if entity_class not in unique_entities:
                unique_entities[entity_class] = set()
            unique_entities[entity_class].add(entity_text)

        stats = {
            "total_documents": total_docs,
            "total_characters": total_chars,
            "total_extractions": len(all_extractions),
            "extractions_per_document": len(all_extractions) / total_docs if total_docs > 0 else 0,
            "class_counts": dict(class_counts),
            "unique_entities": {k: len(v) for k, v in unique_entities.items()},
            "coverage": {
                "person": len(unique_entities.get("person", [])),
                "organizations": len(unique_entities.get("organizations", [])),
                "address": len(unique_entities.get("address", []))
            }
        }

        return stats

if __name__ == "__main__":
    # Example usage
    extractor = LangExtractNERExtractor()

    sample_text = textwrap.dedent(
        """
        Làng cắn chì Ông Trương Hồi và bà Phan Thị Bến vẫn cắn chì từng ngày, bất chấp lời khuyến cáo của bác sĩ Làng đầu tiên chúng tôi tìm về là làng Ngư Mỹ Thạnh cạnh phá Tam Giang, thuộc xã Quảng Lợi, huyện Quảng Điền, cách TP Huế chừng 30km về phía bắc. 
        Ông trưởng thôn Trần Vọng vừa dẫn chúng tôi đi quanh xóm, vừa nói thao thao bất tuyệt về nghề chài lưới quê ông. Theo lời ông Vọng, cả làng Ngư Mỹ Thạnh trước đây đều là dân vạn đò, được chuyển lên bờ, tất thảy sống dựa vào mặt phá với nghề lưới bạc.
        Đi quanh một vòng để chứng minh sự “hưng thịnh” của nghề (vì hầu hết nhà nào cũng có nghề làm lưới bạc), ông Vọng dẫn tôi vào nhà ông Đặng Thu - một căn nhà nhìn ra mặt phá. Đầu giờ chiều, hai vợ chồng ông Thu đang hì hục gắn chì, kết phao bên tay lưới bạc ở góc nhà. 
        Tôi thấy cả hai vợ chồng cầm xếp chì lá trên tay, dùng kéo cắt thành từng miếng nhỏ chừng 1cm2. Cả hai người sau đó vốc từng nắm chì cho vào miệng ngậm. Họ vừa dùng tay đưa lưới lên môi, vừa dùng lưỡi lừa từng miếng chì một ra răng cửa. 
        Cả lưỡi lẫn răng cùng cuốn chì vào lưới và sau đó dùng răng đính chì lại thêm lần nữa để chắc hơn sự kết dính của chì và lưới. Xong, họ lại lừa miếng chì khác ra răng cửa... 
        Lâu lâu, ông Thu nhổ một ngụm nước miếng toàn màu đen. Thấy tôi áy náy về việc cắn chì như thế là nguy hiểm, ông Thu nói: “Chì thì có chi mà độc chú? Không làm nghề ni thì biết làm nghề chi đây, trong khi ruộng không có lấy một tấc. Mà cả làng tui ai mà chẳng làm, có hề chi mô”. 
        “Ở trên phố (TP Huế) có bán loại lưới đã có sẵn chì, nhưng đó là hàng công nghiệp, được đính bằng máy theo lối đại trà nên lưới đem về dùng vài ba bữa là phải làm lưới lại. Mà làm chi có loại máy làm tốt hơn miệng mình được chú?” - vừa làm ông Thu vừa nói. 
        Làm lưới bạc như ông Thu thì trong làng ai cũng thường xuyên làm, trước nhất là để dùng cho nhà mình. Sau khoảng một tháng đi bủa vây trên đầm phá, nhà nào cũng phải đem lưới ra sửa, kết chì vào lưới... 
        Chúng tôi tìm về thôn Tân Vinh, xã Vinh Hiền (huyện Phú Lộc), việc ngậm chì tại làng này sôi động không kém. Toàn thôn có 220 hộ thì hầu như nhà nào cũng có nghề lưới bạc. 
        Có rất nhiều vị cao niên như cụ Trần Chữ, 83 tuổi, có thâm niên lên đến hơn 50 năm trong nghề... ngậm chì; vợ chồng cụ Bùi Chúc, 83 tuổi, và cụ Ngô Thị Gạc, 74 tuổi, làm nghề cũng ngót nghét 50 năm. 
        “Không hề chi, chì có chi mà độc. Mọi người ngậm chì cả mấy chục năm ni có hề chi mô...”. Tuy nhiên, cụ Trần Chữ cùng cụ Bùi Chúc cũng như nhiều cụ khác trong thôn có tiền sử ngậm chì lâu năm đều thừa nhận thường xuyên bị một triệu chứng chung nhất là đau bụng, thường thì lâm râm, có khi đau quặn thắt, sức khỏe yếu hẳn, răng lợi của họ đều bị đen thẫm... và luôn trong tình trạng của một người bệnh kinh niên. 
        Thế nhưng không ai biết mình đang có bệnh gì vì không bao giờ họ đi khám. Khi hỏi vì sao không đi khám, các cụ lại bảo không có tiền. Vả lại khám lòi ra nhiều bệnh thì con cái thêm khổ, ăn đã không đủ rồi còn bệnh với tật...! 
        Theo danh sách do Bệnh viện T.Ư Huế cung cấp, chúng tôi tiếp tục đi tìm một số bệnh nhân bị nhiễm độc chì đã được phát hiện và chữa trị. Tất cả đều sống tại hơn 20 xã ven vùng đầm phá. 
        Người đầu tiên chúng tôi gặp là bà Phan Thị Bến, 52 tuổi, trú tại đội Mười, xã Vinh Giang, huyện Phú Lộc. Bệnh án của bà Bến ghi rõ bà nhập viện và chữa trị nhiều lần trong thời gian nhiều năm liền. 
        Lần gần đây nhất vào cuối năm 2004, bà nhập viện với bệnh viêm dạ dày, chướng bụng, đầy hơi, loét sâu, biến dạng hành tá tràng... Trước khi xuất viện, các bác sĩ khoa nội - tiêu hóa BV T.Ư Huế đã không quên nhắc nhở bà đừng ngậm chì nữa. 
        Thế nhưng trước mắt chúng tôi là một phụ nữ với dáng vẻ uể oải, sắc diện tái xanh vì bệnh tật, vẫn ngồi ngậm chì cùng chồng bên tay lưới bạc. “Thì cũng phải làm mà ăn chứ, không làm nghề ni thì biết làm nghề chi?” - bà nói. 
        Thế nhưng chuyện về ông Trương Hồi - chồng bà Bến - mới làm tôi thật sự kinh ngạc. Người đàn ông 57 tuổi này đã có thâm niên gần 40 năm trong nghề cắn chì, đến nay được xem như người cắn chì giỏi nhất trong vùng. 
        Hàm răng của ông Hồi tất cả đều rất ngắn, cụt ngủn, đen thẫm nằm trên bờ lợi cũng thâm đen. “Cắn chì nhiều quá, răng mòn hẳn. Kiểu ni sợ ít năm nữa e không còn răng mà cắn chì nữa mô!” - ông Hồi nói. 
        Do ngậm chì nhanh gấp đôi thiên hạ và làm chắc chắn nên từ nhiều năm nay ông thường xuyên nhận làm thuê lưới bạc cho dân trong vùng. Để kiếm được tiền công 50.000-60.000 đồng/ngày, ông Hồi phải ngậm hơn 1kg chì mỗi ngày để kết vào lưới. 
        Và theo sự nhẩm tính của ông: “Mỗi năm chì vào miệng ông có đến vài tạ”. Không sợ độc? “Cả đội Mười ni vẫn ngậm như thường chứ riêng gì tui mà lo” - ông Hồi nói. 
        Trường hợp tương tự chúng tôi cũng bắt gặp rất nhiều nơi, như ông Bùi Kháng, 55 tuổi, ở Vinh Hiền, bị mổ pôlip túi mật mà nguyên nhân chính do nhiễm chì, nhưng vượt ra khỏi lời khuyến cáo của bác sĩ, vẫn làm lưới, tiếp xúc với chì. Hay như ông Đ.T. (đề nghị giấu tên) ở Quảng Điền, bị viêm loét dạ dày nặng, nguyên nhân chính được xác định do nhiễm chì, vừa chữa bệnh xong và đang trong giai đoạn sức khỏe ổn định thì lại phải ngậm chì hằng ngày để kiếm cái ăn trước mắt... Cái lý của họ thật đơn giản: không làm thì biết lấy gì mà ăn, mà nuôi con?
        """
    )
    entities = extractor.extract_entities(sample_text)
    print("Extracted Entities:", entities)