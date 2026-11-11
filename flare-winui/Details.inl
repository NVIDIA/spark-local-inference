// SPDX-License-Identifier: Apache-2.0
    void property(StackPanel const& body,std::wstring const& name,std::wstring const& value) {
        Grid row;row.ColumnSpacing(16);ColumnDefinition key;key.Width(GridLength{140,GridUnitType::Pixel});row.ColumnDefinitions().Append(key);row.ColumnDefinitions().Append(ColumnDefinition());
        row.Children().Append(label(name,12,true));auto text=label(value.empty()?L"Not reported":value,12);Grid::SetColumn(text,1);row.Children().Append(text);body.Children().Append(row);
    }
    fire_and_forget model_details(ModelItem model) {
        auto lifetime=get_strong();
        ModalScope modal_scope{modal_open_};if(!modal_scope)co_return;
        try {
            auto modal=dialog(model.name);modal.PrimaryButtonText(model.flags&FLARE_MODEL_ACTIVE?L"":model.available()?L"Use model":L"Download");
            modal.DefaultButton(model.flags&FLARE_MODEL_ACTIVE?ContentDialogButton::Close:ContentDialogButton::Primary);
            modal.IsPrimaryButtonEnabled(model.error.empty()&&(!model.available()||can_change_model_));
            detail_model_=model;detail_rows_=StackPanel();detail_rows_.Spacing(12);
            bool hub_only=model.catalog()&&!model.available();
            if(hub_only) {
                property(detail_rows_,L"Repository",model.source);
                if(!model.author.empty())property(detail_rows_,L"Author",model.author);
                if(!model.task.empty())property(detail_rows_,L"Task",model.task);
                if(!model.base_model.empty())property(detail_rows_,L"Base model",model.base_model);
                if(!model.quantization.empty())property(detail_rows_,L"Reported precision",model.quantization);
                if(model.metadata_known&FLARE_MODEL_CONTEXT)property(detail_rows_,L"Published context",count_text(model.context_tokens)+L" tokens");
                if(model.metadata_known&FLARE_MODEL_PARAMETERS)property(detail_rows_,L"Parameters",count_text(model.parameter_count));
                if(!model.hardware.empty())property(detail_rows_,L"Published builds",model.hardware);
                if(!model.license.empty())property(detail_rows_,L"License",model.license);
                if(model.metadata_known&FLARE_MODEL_REVISION_SIZE){property(detail_rows_,L"Revision files",bytes(model.revision_bytes));detail_rows_.Children().Append(label(L"Includes every file in this Hub revision. The selected core download and GPU memory use can differ.",12,true));}
                if(!model.revision.empty())property(detail_rows_,L"Revision",model.revision);
                if(!model.revision_updated_at.empty())property(detail_rows_,L"Last updated",updated_text(model.revision_updated_at));
                if(!model.revision_commit.empty())property(detail_rows_,L"Commit",model.revision_commit);
                detail_rows_.Children().Append(label(L"These details are reported by the Hugging Face repository. Core verification and compatibility checks run before use.",12,true));
            } else detail_rows_.Children().Append(label(L"Inspecting core metadata…",14,true));
            ScrollViewer scroll;scroll.MaxHeight(470);scroll.Content(detail_rows_);modal.Content(scroll);detail_dialog_=modal;detail_open_=!hub_only;detail_fingerprint_.clear();
            if(!hub_only)accepted(flare_app_inspect_model(app_,model.id.data(),model.id.size()));
            auto result=co_await modal.ShowAsync();modal_scope.release(); detail_open_=false;detail_dialog_=nullptr;detail_rows_=nullptr;render_model_cards();
            if(result==ContentDialogResult::Primary&&!closing_) {if(model.available())activate(model.id,model.name);else download_repository(model.source.empty()?model.name:model.source);}
        }catch(hresult_error const& error){detail_open_=false;show_error(std::wstring(error.message()));}
    }
    void update_detail(flare_snapshot const* snapshot,flare_status const& status) {
        if(!detail_open_||!detail_rows_)return;
        flare_model_detail value{};
        if(!flare_snapshot_model_detail(snapshot,&value)||utf8(value.id)!=detail_model_.id) {
            if(status.error.length&&!status.detail_loading){detail_rows_.Children().Clear();detail_rows_.Children().Append(label(wide(status.error),13,true));}
            return;
        }
        compatibilities_[detail_model_.id]=value.compatibility;
        auto fingerprint=wide(value.metadata_status)+wide(value.path)+wide(value.source_revision)+wide(value.publisher)+wide(value.inspection_error)+wide(value.compatibility_reason)+std::to_wstring(value.known_fields)+std::to_wstring(value.compatibility);
        if(fingerprint==detail_fingerprint_)return;detail_fingerprint_=fingerprint;
        detail_rows_.Children().Clear();detail_rows_.Children().Append(label(detail_model_.description.empty()?L"Prepared Flare core":detail_model_.description,14,true));
        property(detail_rows_,L"Source",detail_model_.catalog()?detail_model_.source:L"On this PC");
        if(!detail_model_.revision_updated_at.empty())property(detail_rows_,L"Hub last updated",updated_text(detail_model_.revision_updated_at)+L" · "+detail_model_.revision);
        property(detail_rows_,L"Publisher",wide(value.publisher));
        property(detail_rows_,L"Core checks",wide(value.metadata_status));
        property(detail_rows_,L"Core size",detail_model_.size_known?bytes(detail_model_.size):L"Not reported");
        property(detail_rows_,L"Prepared context",value.known_fields&FLARE_DETAIL_CONTEXT?std::to_wstring(value.context_tokens)+L" tokens":L"Not reported");
        property(detail_rows_,L"Concurrency",value.known_fields&FLARE_DETAIL_CONCURRENCY?std::to_wstring(value.max_concurrency):L"Not reported");
        property(detail_rows_,L"Architecture",wide(value.architecture));
        property(detail_rows_,L"Prepared ranks",value.known_fields&FLARE_DETAIL_RANKS?std::to_wstring(value.rank_count):L"Not reported");
        property(detail_rows_,L"Recorded allocations",value.known_fields&FLARE_DETAIL_MEMORY?bytes(value.prepared_memory_bytes):L"Not reported");
        if(value.memory_note.length)detail_rows_.Children().Append(label(wide(value.memory_note),12,true));
        detail_rows_.Children().Append(label(wide(value.compatibility_reason),13,true));
        if(value.inspection_error.length)detail_rows_.Children().Append(label(wide(value.inspection_error),12,true));
        Expander technical;technical.Header(box_value(L"Core files and revision"));technical.HorizontalAlignment(HorizontalAlignment::Stretch);StackPanel fields;fields.Spacing(9);
        property(fields,L"Path",wide(value.path));property(fields,L"Version",detail_model_.version);property(fields,L"Source revision",wide(value.source_revision));property(fields,L"Official source",wide(value.official_source));technical.Content(fields);detail_rows_.Children().Append(technical);
        if(value.can_remove_reference){
            Button locate;locate.Content(box_value(L"Locate core…"));locate.Click([this](auto const&,auto const&){
                try {locate_reference(detail_model_);}
                catch(hresult_error const& error){show_error(std::wstring(error.message()));}
            });detail_rows_.Children().Append(locate);
            Button remove;remove.Content(box_value(L"Remove library reference"));remove.Click([this,id=detail_model_.id](auto const&,auto const&){accepted(flare_app_remove_reference(app_,id.data(),id.size()));if(detail_dialog_)detail_dialog_.Hide();});detail_rows_.Children().Append(remove);
            detail_rows_.Children().Append(label(L"Removing a reference leaves the original files in place.",12,true));
        }
        if(detail_model_.available()&&value.compatibility==FLARE_COMPATIBILITY_INCOMPATIBLE)detail_dialog_.IsPrimaryButtonEnabled(false);
    }
    void locate_reference(ModelItem const& model) {
        com_ptr<IFileOpenDialog> picker;check_hresult(CoCreateInstance(CLSID_FileOpenDialog,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(picker.put())));
        check_hresult(picker->SetTitle(L"Locate prepared core"));DWORD options{};check_hresult(picker->GetOptions(&options));check_hresult(picker->SetOptions(options|FOS_FORCEFILESYSTEM|FOS_FILEMUSTEXIST));
        auto result=picker->Show(hwnd_);if(result==HRESULT_FROM_WIN32(ERROR_CANCELLED))return;check_hresult(result);
        com_ptr<IShellItem> item;check_hresult(picker->GetResult(item.put()));PWSTR path{};check_hresult(item->GetDisplayName(SIGDN_FILESYSPATH,&path));std::unique_ptr<wchar_t,decltype(&CoTaskMemFree)> owner(path,CoTaskMemFree);
        auto operation=flare_app_locate_reference(app_,model.id.data(),model.id.size(),reinterpret_cast<uint16_t*>(path),wcslen(path));
        accepted(operation);if(operation&&detail_dialog_)detail_dialog_.Hide();
    }
